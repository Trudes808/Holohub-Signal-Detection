# Data-reduction & compute evaluation

Quantifies two things about mask-driven **real-time** signal collection, as hand-calcs over the
attenuation-sweep ("wired") captures:

1. **Data reduction** — storage saved vs. naively saving every sample, and the *fidelity cost*.
2. **Compute** — what each detector costs to run in real time (GPU FLOPs + memory) vs. its accuracy.

`data_reduction_eval.ipynb` is self-documenting (methodology + every formula inline); this README is
the short version. Kernel: **Python (yolo)** (pure analysis — reads masks + tables + `compute_table.csv`;
loads no models).

## Baseline
Captures are 245.76 MHz complex, `cf32` (8 B/sample) → **SAVE-ALL = 7.08 TB/hour** (flat, SNR-independent).

## Reduction strategies (Figure 1)
| strategy | formula | note |
|---|---|---|
| Save-all | `rate·bytes·3600` = 7.08 TB/hr | flat reference |
| Time-slicing | `SAVE_ALL · timeslice_frac` | keep time-blocks with ≥1 detected cell; block size is a knob (~42 µs default). Modest (~1.3–2×). |
| Resample+filter | `SAVE_ALL · Σ(region_bw·region_time)/(full)` | **big win (~20–200×)**; the real operator is a **collaborator HOOK** (`resample_filter_bytes_per_hour`). The notebook plots a projection from raw-mask TF-coverage until that lands. |

## Fidelity: signal retention
`retention = |GT_time_rows ∩ detector_time_rows| / |GT_time_rows|` — of a signal's time, the fraction
the detector keeps. Reduction is only meaningful paired with retention (else "saved more" = "missed more").
**Coherent Power is the baseline to beat** — it is *supposed* to emit a mask even if unoptimized.

## Figures
- **Fig 1** — bytes/hr (log) vs attenuation: save-all vs time-slice vs resample+filter (projected).
- **Reduction vs retention** — the honest trade-off (path per detector across 0→60 dB).
- **Reduction × retention × accuracy** — trajectory-bubble; color = pixel-IoU of the kept mask.
- **Fig 2** — detection IoU vs attenuation (line per model), paired with the compute table.
- **Compute** — GPU memory + real-time factor bars; **TB-saved/hr** benefit view.

## Compute (Figure 2 cost side)
`compute_table.csv` from `yolo_training/src/measure_compute.py`: FLOP estimates (ViT-B-scaled for DINO,
ultralytics for YOLO, FFT for coherent), **measured GPU memory** for the offline models (fine-tuned
DINO M1/M2, YOLO26m), and real-time throughput vs the **938 tiles/s** needed to keep 245.76 MS/s.
Headline: single-GPU real-time factor ≈ 0.09× (DINO), 0.17× (YOLO26m), ≫1 (coherent) — the learned
detectors need multi-GPU/downsampling; coherent is trivially real-time. Container-only detectors
(coherent, zero-shot) are FLOP/mem *estimates*.

## Regenerate
```bash
conda activate dinov3 && python ../../yolo_training/src/measure_compute.py   # -> compute_table.csv
conda activate yolo   && jupyter nbconvert --to notebook --execute data_reduction_eval.ipynb
```
Knobs: `DS_NFRAMES` (frames/stem, default 120), `DS_SWEEP` (mask root), time-slice block size in `timeslice_frac`.
Data (`compute_table.csv`, generated `*.png`, `*_table.csv`) is gitignored; the `.ipynb` + `.py` + this README are tracked.

## Open items
- Fill the `resample_filter_bytes_per_hour` hook with the real rational-resample/frequency-filter operator (collaborator).
- Real wall-clock for coherent + zero-shot needs a container replay (currently FLOP/mem estimates).

## Adding coherent-power + zero-shot DINO to the LIVE-OTA figure (Figure 3)

Those two are container-only C++ detectors, so their live masks need a container batch run
(needs `sudo docker`; **no rebuild** — both detectors already ship in the demo container).
Run it so the output lands at the path the measurement auto-reads
(`/tmp/usrp_spectrograms/batch_eval/live_ota`):

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
python3 run_batch_offline_eval.py \
    --captures-dir ~/captures/live_data/sigmf_out \
    --detectors coherent_power cuda_dino \
    --run-id live_ota \
    --output-root /tmp/usrp_spectrograms/batch_eval/live_ota \
    --repack-masks --no-post
```
Produces `/tmp/usrp_spectrograms/batch_eval/live_ota/{coherent_power,cuda_dino}/<test_*>/mask_arrays/…`.
(Live data is unlabeled → GT/coverage scoring is empty, but the detector **masks** — all we need — are produced.)

Then refresh Figure 3 — the two hatched "pending" bars become real automatically:
```bash
conda activate dinov3 && python ../../yolo_training/src/measure_live_saving.py 50   # auto-reads live_ota masks
conda activate yolo   && jupyter nbconvert --to notebook --execute data_reduction_eval.ipynb
```
`measure_live_saving.py` checks `/tmp/usrp_spectrograms/batch_eval/live_ota/{coherent_power,cuda_dino}`;
if present it computes their OTA time-slice/coverage and drops them into `live_data_saving.csv` (status `container`).

## Measured resample+filter (signal_snipper) — offline

The resample+filter footprint in Figure 3c is **measured from the real `signal_snipper` operator**
(not a projection). The snipper (frequency mode) cuts each detected signal out of the wideband IQ,
mixes to baseband, low-passes to the signal bandwidth, and decimates; `sigmf_file_sink` writes the
result as SigMF. Its output bytes, scaled to an hour of capture, are the measured footprint.

Produce measurements (container / lab-admin sudo; `CONTAINER_NAME` must match the built container):

```bash
cd applications/usrp_wideband_signal_detection
# coherent_power masks drive the snipper (default); add DETECTOR=cuda_dino for zero-shot DINO masks
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_offline_snipper.sh
# or explicit captures:
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_offline_snipper.sh /path/a.sigmf-data ...
```

This replays each capture through `config_signal_snipper_single_channel.yaml` (offline mode injected)
and writes snippets to `/tmp/usrp_spectrograms/snippets_eval/<detector>/<stem>/snippets/*.sigmf-data`
(host-visible via the `/workspace/spectrograms` mount). Re-run the notebook: the notebook's
`resample_filter_bytes_per_hour(detector, stem, capture_sec)` reads those SigMF sizes and the
"pending snipper" bars in Figure 3c fill in with measured values. Knobs: `DS_SNIP_ROOT`,
`DS_OTA_STEMS`, `DS_OTA_CAP_SEC` (default 3.0 s/file for the 5.9 GB OTA captures).

# soft_label_pipeline — recordings → detector → soft labels

A self-contained pipeline that takes **recordings** (single files or whole directory trees) and a
**detector**, runs the detector to produce per-frame masks, and emits soft labels:

- **`<stem>_softlabel.sigmf-meta`** (+ `<stem>_softlabel.sigmf-data`) — a SigMF recording whose
  `annotations[]` are the detector's `detected_waveform` boxes on the recording's own timeline
  (`core:sample_start/count`, `core:freq_lower/upper_edge`), written to `--out-dir` mirroring the input
  tree. The `.sigmf-data` is a symlink to the original samples (or their converted copy for formats that
  are not raw IQ), so each pair is a complete, valid SigMF recording. Always produced.
- **decimated IQ pairs** — `snippets/*.sigmf-{data,meta}`, one recording per detection, each **cut in
  time and decimated in frequency** to its bandwidth (mix → low-pass → decimate). Optional, container
  detectors only.

`--outputs {softlabel_meta | iq | both}` selects one, the other, or both (`snipped_meta` is still accepted
as the old name of `softlabel_meta`).

## Quickstart: the released fine-tuned DINO models, no container
```bash
# from this folder; needs Python with torch + numpy + scipy + pyyaml (h5py for .h5/.hdf5/MATLAB v7.3 inputs)
python3 soft_label_pipeline.py --input /data/recordings --recursive \
    --detector dino_finetuned_m12s --out-dir /data/labels
#   -> /data/labels/<same sub-folders>/<stem>_softlabel.sigmf-{meta,data} + /data/labels/softlabel_index.csv
python3 soft_label_pipeline.py --input a.sigmf-meta b.bin c.npy --rate 25e6 --detector dino_finetuned_m13s --out-dir out/
```
The model file comes from `dino_fine_tuning/weights/` and is downloaded there from the GitHub release when
missing. Masks and other working files go under `--output-root` (default
`/tmp/usrp_spectrograms/soft_label_pipeline`; keep it off `/home`). Runs resume: a recording whose masks
exist for the same model and settings is not recomputed (`--force` redoes it).

## 1. Detectors & prerequisites
Choose one with `--detector`:

| tier | detectors | how masks are made | needs |
|---|---|---|---|
| **torchscript** | `dino_finetuned_m12s`, `dino_finetuned_m13s`, `dino_finetuned_m7s`, `dino_finetuned_ts` (+ `--ts-model x.ts`) | `dino_ts_masks.py`: the TorchScript export + the Python twin of the operator's front end (`dino_fine_tuning/src/frontend.py`, `rates.py`) | PyTorch (GPU recommended, CPU works); **no container, no sudo** |
| **trained** | `coherent_power`, `cuda_dino` | container binary, directly from IQ | demo container (§3) |
| **baselines** | `3dB_power`, `blob_detection` | `run_baseline_offline.py`, mirroring a trained run | container (for the ref run) + numpy/scipy (CPU) |
| **ml** | `yolo`, `dino_finetuned`, `dino_finetuned_m1`, `yolo26s` | `run_ml_detectors_offline.py`, mirroring a trained run | container (ref) + **GPU** + conda `dinov3` (torch, ultralytics) + model weights + dinov3 repo |

**Torchscript tier.** Each name is defined by the live config of that model
(`config_live_v3_dino_ft_{m12s,m13s,m7s}.yaml`): model file, decision threshold (0.6), front-end
settings, invalid-frame guard and the snipper's clustering gates (`min_box_pixels` 256,
`merge_gap_rows` 16, `merge_gap_cols` 8), so the labels follow the same settings as the live demo. Which
model: **M12-S** recalls weak and low-SNR emissions best, **M13-S** has the fewest false positives but
drops some weak flat carriers, **M7-S** is what the live demo deploys (`../../../README.md` §6 has the
details). `dino_finetuned_ts --ts-model <file>.ts` runs any other `dino_fine_tuning` export with the M12-S
settings and the threshold from its `.meta.json`.

What `dino_ts_masks.py` reproduces from the C++ `finetuned_dino_detector`: frames of 512 rows × the app's
FFT size for the sample rate, the robust-normalised front end, two 256 × 1024 tiles, sigmoid ≥ threshold,
the 1024-column mask widened to the FFT size with the operator's nearest-column index, and the
invalid-frame guard (a frame denser than `max(min_occupancy, k × running baseline)` is emitted empty). It
writes the same mask contract as the container (`mask_arrays/mask_ch0_f{N}_{rows}x{cols}.packed.npz` +
`frame_manifest.csv`), so `snip_annotations.py` and `--mask-root` work unchanged. Recordings that do not
fill whole frames: a clip shorter than one frame is repeated cyclically to fill it and a recording's tail
is covered by one frame aligned to its end; mask rows that are not new real samples are zeroed, so nothing
is reported twice or outside the recording. Clips shorter than one FFT row (1024 samples) are skipped.
Sample rates outside the models' training range (5–500 MS/s) still run, with a note.

`--no-frame-guard` keeps frames the guard would empty: the guard is there for corrupted live-ingest
frames, and a clean recording can be legitimately that dense. `--device cpu` works on machines without a
GPU (the released exports were traced on a GPU, so a `<stem>.cpu.ts` copy with the device constants
pointed at the CPU is written next to the weights). `--max-seconds S` labels only the first S seconds of
each recording.

**Dependency chain (baselines, ml):** baselines and ML detectors do *not* process raw IQ on their own —
they reuse a trained detector's frame grid, ground truth, and `frame_manifest.csv`. So when you pick a
baseline/ML detector, the pipeline first **seeds the reference trained detector** (`--ref-detector`,
default `cuda_dino`) on every capture, then runs the chosen one. That means even a `blob_detection` run
needs the container.

**ML weights** are configured in `../../baseline_comparisons/comparison_config.yaml` (`ml_detectors:`)
— `dino_finetuned` = M2, `dino_finetuned_m1` = M1, `yolo` = YOLO26-m, `yolo26s` = YOLO26-s. Point at
your dinov3 repo with `--dinov3-repo` (default from the config). These are gitignored, GPU-resident.

**Adding a new ML detector:** the pipeline reads the ML detector list *dynamically* from
`comparison_config.yaml`'s `ml_detectors:` block, so a new model auto-enables here with no edit to
`soft_label_pipeline.py` — add an entry (a `kind: {yolo|dino_finetuned}` + its `ckpt`/paths, matching
what `run_ml_detectors_offline.py` expects) and run `--detector <your_new_name>`. Use `--config` to
point at a different comparison config entirely. A new TorchScript export needs no entry: use
`dino_finetuned_ts --ts-model`, or add a name to `TS_DETECTORS` in `dino_ts_masks.py` with its live config.

## 2. Inputs
`--input` takes files and/or directories (`--recursive` descends; `--glob PATTERN` filters file names;
`--limit N`); `--waveform-dir DIR` is the same as `--input DIR`. Every recording is first written as the
`<stem>_softlabel.sigmf-{meta,data}` pair by `to_sigmf.py`, which reads:

| container | how the sample rate, centre frequency and sample format are found |
|---|---|
| SigMF 1.x (`.sigmf-meta`) | from the metadata, any SigMF sample format (cf32, cf64, ci16, ci8); frequencies written in MHz are fixed, `wfgt:rf_center_freq_hz` beats a baseband `core:frequency` |
| SigMF pre-1.0 (`_metadata` wrapper) | normalised; a wrong `core:datatype` is corrected by a preset (ORACLE) |
| raw binary (`.bin .dat .raw .iq .cfile .cf32 .cs16 .cu8 …`) | `--rate/--fc/--dtype`, else a JSON sidecar (`<stem>.json`, or a `metadata.json` listing the file; for a symlink also next to its target), else a dataset preset, else the extension; the sample format is guessed from the bytes as a last resort (with a note) |
| `.npy`, `.npz` | complex arrays as they are; real arrays with I/Q in a last axis of 2, interleaved (signed integers) or planar (`--iq-layout`); `fs`/`fc` scalars in an `.npz` |
| `.mat` (v5; v7.3 needs h5py) | the largest complex variable (or `--var`); the rate from `Fs`/`fs`/`sample_rate` |
| `.h5`, `.hdf5` (h5py) | the largest complex dataset (or `--var`); rate/centre from attributes |
| `.pkl`, `.pickle` | numpy arrays in dicts/lists only, through a restricted unpickler that refuses anything else |
| `.wav` | 2-channel IQ (left I, right Q); centre from a `<n>Hz` token in the file name |

Arrays with leading dimensions (clips × samples, channels × pulses × samples) become one recording with one
capture segment per clip (`--index 0` picks channel 0 first, `--max-segments N` the first N clips).
`python3 to_sigmf.py --list-presets` prints the presets for the datasets under `~/Datasets` (DroneDetect,
T-PRIME, ORACLE, GENESYS CBRS, x410 HCRO, WiSig, RadarML, RadSeg, RFChallenge); command-line values beat
presets, and every assumption made lands in the output's `argus:conversion_notes`. RadarML, RadSeg and
RFChallenge do not document their sample rate: pass `--rate`. Folders named `masks/`, `mask_arrays/`,
`overlays/`, `panels/`, `figures/` and the pipeline's own `*_softlabel.sigmf-*` files are never taken as
inputs. The source files are never modified; the source's own annotations are not copied into the soft
label (their count and the source path are in the global block).

`to_sigmf.py` also runs on its own (`python3 to_sigmf.py <inputs> --out DIR [--recursive] [--cf32]`);
`--cf32` writes real cf32_le files, which is what the container tiers need.

**Container tiers** (trained, baselines, ml) keep their original input contract: one flat
`--waveform-dir` of `cf32_le` SigMF pairs (`<name>.sigmf-data` + `<name>.sigmf-meta` with
`global.core:sample_rate` and `captures[0].core:frequency`). Any sample rate works (the container reads
`core:sample_rate` from the meta). The lab captures are 245.76 MSps; the OTA captures are 500 MSps.

## 3. Docker container setup (container tiers only)
Mask generation and the IQ snipper of the trained / baseline / ml tiers run inside the demo container (the
pipeline calls `run_cuda_dino_offline_file.py`, which does `sudo docker exec …`). From the app root
(`applications/usrp_wideband_signal_detection`):
```bash
# first-time container bring-up (image already built elsewhere):
sudo env SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh
# after any code/config change to the app:
sudo ./bash_scripts/rebuild_demo_container_app.sh
```
- Container/image identity lives in `bash_scripts/container_env.sh`; override per run with
  `--container-name` (or `CONTAINER_NAME=…`). Default here: `usrp_x410_sig_det_bqn82`.
- The container must have the `mask_replay_detector` operator compiled in (a `rebuild_demo_container_app.sh`).
- `sudo docker` must be passwordless for the account (the wrappers rely on it); run the Python
  pipeline **without** `sudo`.
- **Mounts / paths:** the container maps `/home/bqn82/captures → /workspace/captures`,
  `/tmp/usrp_spectrograms → /workspace/spectrograms`, and the repo → `/workspace/holohub`. Your
  `--waveform-dir` and `--output-root` should sit under one of those host roots so files are read/written
  in place (no copy). A `--waveform-dir` elsewhere still works but the driver stages a scratch copy of
  each capture (slow for large files).

## 4. Usage examples
```bash
# released fine-tuned DINO, a whole tree of any format, labels into a folder of your choice (no container):
python3 soft_label_pipeline.py --input ~/Datasets/DroneDetect_V2 --recursive \
    --detector dino_finetuned_m12s --out-dir ~/labels/dronedetect_m12s

# a few files, fewer false positives, CPU only, first 2 s of each:
python3 soft_label_pipeline.py --input rec1.sigmf-meta rec2.bin --rate 30.72e6 --fc 915e6 \
    --detector dino_finetuned_m13s --device cpu --max-seconds 2 --out-dir ./labels

# arrays with an undocumented rate (one clip per segment), first 100 clips:
python3 soft_label_pipeline.py --input ~/Datasets/RadSeg/RadSegTest/radseg_iq.hdf5 --rate 10e6 \
    --max-segments 100 --detector dino_finetuned_m12s --out-dir ./labels

# trained detector, both outputs, one capture (container):
python3 soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --glob "attenuation_dB_30.sigmf-data" --detector coherent_power --outputs both

# fine-tuned DINO (M2), soft-label meta only, whole folder (container + GPU + weights):
python3 soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --detector dino_finetuned --outputs softlabel_meta

# reuse pre-computed masks (skip generation), IQ pairs only:
python3 soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --glob "attenuation_dB_45.sigmf-data" --detector coherent_power \
    --mask-root /tmp/usrp_spectrograms/all_detectors --outputs iq

# tighter gate (drop tiny detections) via the snipper selectivity flags:
python3 soft_label_pipeline.py --waveform-dir DIR --detector cuda_dino --outputs both \
    --min-box-pixels 256 --min-bandwidth-hz 75000 --min-duration-s 0.001 --min-mask-bandwidth-hz 75000
```

## 5. Outputs
```
<out-dir>/                                   (default <output-root>/softlabels)
  <sub-folders of the input>/<stem>_softlabel.sigmf-meta   detected_waveform annotations + provenance
  <sub-folders of the input>/<stem>_softlabel.sigmf-data   symlink to the samples (or their converted copy)
  softlabel_index.csv                        one row per recording: source, soft label, rate, centre, frames,
                                             guard-emptied frames, detections, notes (or why it was skipped)
<output-root>/
  masks/<detector>/<key>/                    mask_arrays/ + frame_manifest.csv (+ ts_model.json for the
                                             torchscript tier; + gt_masks + gt_annotations for the container);
                                             <key> = <stem>, prefixed by the input sub-folders joined with "__"
  iq/<stem>/snippets/*.sigmf-{data,meta}     decimated IQ per detection   (--outputs iq|both, container tiers)
  config_snip_generated.yaml                 the mask-replay/snipper config the IQ step used
```
Each annotation: `core:sample_start`, `core:sample_count`, `core:freq_lower_edge`, `core:freq_upper_edge`
(absolute RF when the centre is known), `core:label: "detected_waveform"`, `wfgt:detector`,
`wfgt:frame_number`. The global block records the source file and format, every conversion assumption
(`argus:conversion_notes`), `argus:soft_label_detector` and the clustering gates used.

## 6. Key options
- inputs/outputs: `--input PATH…` / `--waveform-dir DIR` · `--recursive` · `--glob PATTERN` · `--limit N` ·
  `--out-dir DIR` · `--output-root DIR` · `--outputs {softlabel_meta,iq,both}`
- `--mask-root DIR` — reuse existing `<DIR>/<detector>/<key>/mask_arrays` instead of generating.
- torchscript: `--ts-model x.ts` · `--threshold` · `--device {cuda,cpu}` · `--max-seconds` · `--force` ·
  `--offline` (never download weights) · `--no-frame-guard`
- reading inputs: `--rate` · `--fc` · `--dtype` · `--iq-layout` · `--var` · `--index` · `--max-segments` · `--copy`
- selectivity: `--mode {frequency,time_only}`, `--min-box-pixels`, `--min-bandwidth-hz`,
  `--min-duration-s`, `--min-mask-bandwidth-hz` (snipper); `--merge-gap-rows/-cols` (annotations).
  Defaults: the torchscript model's live snipper gates (256 / 16 / 8), otherwise 256 / 16 / 80.
- container/wiring: `--container-name`, `--captures-root`, `--ref-detector`, `--detector-config`,
  `--config` (baseline/ML params), `--dinov3-repo`.

Note: the `<stem>_softlabel.sigmf-meta` uses the size+merge clustering (`min_box_pixels` + merge gaps);
the IQ snipper additionally applies the bandwidth/duration gates. At the default (256 px, no bw/dur
gate) the two agree one-to-one.

## 7. Validation of the torchscript tier (2026-11-09, DGX Spark GB10)
- **Same masks as the fine-tuning tools:** before widening, the masks equal
  `dino_fine_tuning/src/run_ts_on_sigmf.py` bit for bit (0 of 24.1 M pixels differ, M12-S, N320 FM capture,
  46 frames). Recordings processed later in the same Python process can differ by ~0.002 % of pixels
  (GPU kernel selection inside the bf16 model; the same frame run twice in one process also varies).
- **Same as the container:** M7-S on `test_1` (245.76 MS/s, 10240-column masks) against the container's
  own M7-S masks: 140 / 140 frames on the same sample grid, 0.05 % of pixels differ (separate C++/cuFFT
  front-end implementation), per-frame IoU median 0.978 (min 0.951), mean occupancy 2.222 % vs 2.221 %;
  4294 vs 4271 annotations (the extra ones mostly from the tail frame the container drops).
- **CPU:** masks agree with the GPU at IoU 0.985 (bf16 kernels differ).
- **Formats:** `test_to_sigmf.py` (synthetic round trip of every container, no GPU) passes; one sample of every
  dataset under `~/Datasets` converted and labelled; every output validates with the `sigmf` package.
- **Speed** (GPU, after warm-up, reading included): ~21 ms per frame at 10–25 MS/s (1–2.4× real time),
  ~107 ms per frame at 245.76 MS/s (0.2× real time, 42 MB read per frame); CPU ~0.6 s per frame at 24.6 MS/s.
- On unified-memory machines (DGX Spark) a page cache full of recording data makes CUDA allocations fail
  although the memory is reclaimable; the pipeline drops the pages it has read, and
  `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` clears a cache filled by something else.

## 8. Files here
- `soft_label_pipeline.py` — the orchestrator (this pipeline).
- `dino_ts_masks.py` — masks from the TorchScript DINO-FT exports, no container (also runs on its own).
- `to_sigmf.py` — any IQ format → SigMF pairs (the input stage; also runs on its own).
- `snip_annotations.py` — clusters masks → `<stem>_softlabel.sigmf-meta` (adapted from `snip_eval/`).
- `materialize_npy.py` — unpacks `.packed.npz` masks → `.npy` for the replay snipper (copied).
- `test_to_sigmf.py` — round-trip test of the converter on synthetic files in every format.

The heavy detector-generation tools live in the app (`run_cuda_dino_offline_file.py`,
`baseline_comparisons/run_{baseline,ml_detectors}_offline.py`, `dino_fine_tuning/src/frontend.py`) and are
called in place.

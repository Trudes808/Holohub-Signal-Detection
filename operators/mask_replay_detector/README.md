# mask_replay_detector

A drop-in "detector" that **replays precomputed masks** into the signal-detection
graph so the C++ `signal_snipper` can snip **any** detector's masks — including the
Python-only fine-tuned models (yolo, dino_finetuned) and the CV baselines
(3dB_power, blob_detection) — without those detectors existing as C++ operators.

## What it does
It has the same ports as `cuda_dino_detector`:
- **`in`** — the spectrogram tuple. Used *only* for message metadata
  (`fft_emitted_frame_number` + the `offline_source_*` IQ-offset fields); the tensor
  payload is ignored.
- **`mask_out`** — `DetectorMaskMessage`. For each frame it loads
  `${mask_dir}/mask_ch{channel}_f{frame_number}_{H}x{W}.npy` (C-order uint8), puts it
  in `pixels` (host), copies `frame_number` + IQ offsets from the input metadata, and
  emits. `device_pixels` is left null — the snipper reads host `pixels` when there is
  no device buffer.

Because the port signature matches a detector, it slots straight into the existing
`add_flow(spectrogram, detector)` → `add_flow(detector, snipper.mask_in)` wiring; the
only change is one `DetectorAdapter` entry (`mask_replay`) + a config block.

## Use (offline)
Register `--detector mask_replay` with a config whose `mask_replay_detector.mask_dir`
points at a detector's `mask_arrays/` dir (in the container path space), with
`pipeline.enable_signal_snipper: true`. The graph becomes
`source → fft → spectrogram → mask_replay(masks) → snipper → sigmf_file_sink`, writing
real snipped SigMF. See `config_mask_replay_snip_single_channel.yaml`.

## Params
- `mask_dir` (string) — directory of precomputed masks to replay.
- `channel` (int, default 0) — channel index in the mask filename.
- `emit_stride` (int, default 1) — emit every Nth frame (match the producing run).
- `num_channels` (int, default 1).

## Notes
- Drain / partial frames emit no mask (mirrors `cuda_dino_detector`).
- `.packed.npz` masks (from the Python ML drivers) must be unpacked to `.npy` first.
- Missing mask for a frame → emits an all-zero mask at the last-seen geometry (warns).

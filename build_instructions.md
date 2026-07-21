# Build instructions — native signal-segmentation detector operators (FT-DINO, YOLO26)

**Goal.** Run the offline-only models (fine-tuned DINO M1/M2, YOLO26 s/m) as **native C++ detector
operators** inside the container so `signal_snipper` can produce masks for *all* models (today only
`coherent_power` + `cuda_dino` can be snipped). Native ops — not a mask-replay shim — because
**real-time evaluation** is the eventual goal.

**Locked decisions:** new *dedicated* operators (don't edit the 9k-line `cuda_dino_detector`);
**TorchScript** backend for every model (same as the DINO ops); **FT-DINO first** as the reference
pattern, then YOLO26.

## READMEs / references to build an operator
- `operators/README.md` — HoloHub operator conventions + contributing guide.
- `operators/template/` — scaffold for a new operator (copy as the starting point).
- **`operators/cuda_dino_detector/`** — the closest reference: a DINO detector that emits
  `DetectorMaskMessage`. Model the new operator on its layout
  (`{name}.cu/.hpp`, `*_torch_helpers.cpp/.hpp`, `*_types.hpp`, `CMakeLists.txt`, `metadata.json`,
  `README.md`) and its TorchScript load/run machinery.

## Turnkey scripts (this repo)
`APP=applications/usrp_wideband_signal_detection/bash_scripts`
| script | what it does | where / who |
|---|---|---|
| `dino_fine_tuning/export_finetuned_models.sh` | export FT-DINO M1+M2 → TorchScript (parity-checked) | host, `dinov3` env — **done/verified** |
| `yolo_training/export_yolo_models.sh` | export YOLO26 s+m → TorchScript | host, `yolo` env — **done/verified** |
| `$APP/build_all_detector_operators.sh` | one rebuild → compiles **all** wired operators | lab-admin (docker/sudo) |
| `$APP/build_finetuned_dino_operator.sh` | rebuild (FT-DINO only) | lab-admin (docker/sudo) |
| `$APP/run_all_detectors.sh` | run **all 6** detectors offline → masks (skips unwired ones); `--repack` packs masks for the notebook | lab-admin (docker/sudo) |
| `applications/usrp_wideband_signal_detection/repack_offline_masks.py` | pack existing `.npy` masks → `.packed.npz` across a tree (notebook/snipper format) | lab-admin (sudo; outputs are root-owned) |
| `$APP/test_finetuned_dino_offline.sh` | run FT-DINO offline + IoU vs Python reference masks | lab-admin (docker/sudo) |

---

## The contract every native detector must satisfy
Emit `holoscan::ops::DetectorMaskMessage` (in
`applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp`):
```cpp
struct DetectorMaskMessage {
  std::vector<uint8_t> pixels;          // binary mask, height*width
  std::shared_ptr<uint8_t> device_pixels;
  int width, height, channel;
  uint64_t frame_number;                // COPY from the input frame (mask<->IQ alignment)
  uint64_t file_offset_complex, data_end_complex, frame_end_complex,
           complex_samples_read, complex_samples_padded;   // COPY from the input frame
};
```
Copy the `frame_number`/IQ-offset fields straight from the incoming frame (like the existing
detectors) so the snipper matches each mask to its IQ frame.

---

## FT-DINO — step 1 (DONE + verified): TorchScript export
`dino_fine_tuning/src/export_finetuned_torchscript.py` exports `DinoSegmenter` (DINOv3 ViT-B/16
backbone + trained SegHead). Verified **TorchScript == eager exactly** (max|logit diff| = 0, 100% mask
agreement) for both models. Run it via the wrapper:
```bash
conda activate dinov3
cd ~/Holohub-Signal-Detection/dino_fine_tuning
./export_finetuned_models.sh
```
Artifacts (git-ignored): `dino_fine_tuning/weights/finetuned_dino_m{1,2}.ts` (350 MB) + `.meta.json`.
The repo is mounted at `/workspace/holohub`, so the operator's `model_script_path` is:
```
finetuned_dino     -> /workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m1.ts   (threshold 0.45)
finetuned_dino_m2  -> /workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m2.ts   (threshold 0.85)
```
I/O contract: **input** `float[B,1,256,1024]` in [0,1] (dB spectrogram, 256-row tiles); **output**
`logits[B,1,256,1024]`; **post-proc (in the operator)** `sigmoid(logits) >= threshold` → mask, stitch.

## FT-DINO — step 2 (TODO — code, needs container to compile): the operator
Create `operators/finetuned_dino_detector/` (copy `operators/template/`, model on
`operators/cuda_dino_detector/`):
1. **Front-end (C++):** build the dB spectrogram at `nfft=1024` from the frame IQ (reuse the FFT path /
   the `frames_to_db` math in `dino_fine_tuning/src/finetuned_infer.mask_for_iq`), normalize
   `clamp((db - db_vmin)/(db_vmax - db_vmin), 0, 1)`, split into 256-row tiles (pad last).
2. **Inference:** run the TorchScript module (reuse `cuda_dino_detector`'s torchscript load/run helpers;
   `model_script_path` from config).
3. **Post-proc:** `sigmoid(logits) >= threshold` → per-tile uint8 → stitch → emit `DetectorMaskMessage`
   (copy `frame_number` + IQ offsets from the input frame).
4. `finetuned_dino_detector/CMakeLists.txt` + `metadata.json` (mirror cuda_dino_detector).
5. **Config block** `finetuned_dino_detector:` with `model_script_path`, `threshold`, `tile_rows: 256`,
   `nfft: 1024`, `db_vmin`, `db_vmax` (from `.meta.json`).
6. **Wire into the app** — `applications/usrp_wideband_signal_detection/CMakeLists.txt`: add
   `holoscan::ops::finetuned_dino_detector` to the `DEPENDS OPERATORS` list + `target_link_libraries`,
   and the operator dir to the operators path list (mirror the `cuda_dino_detector` lines).
7. **Register** two `DetectorAdapter` entries (`finetuned_dino`, `finetuned_dino_m2`) in
   `run_offline_cuda_detector_eval.cpp` (and `main.cpp` for live) — same op, different
   `model_script_path`/`threshold`.

## FT-DINO — step 3 (lab-admin): build + test
```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/build_finetuned_dino_operator.sh
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/test_finetuned_dino_offline.sh
```
The test script runs `--detector finetuned_dino` offline and compares its masks to the Python reference
(`notebooks/yolo_evals/sweeps/sweep_all/finetuned_dino`) — expect high IoU. Then the snipper runs for
FT-DINO via a snipper config with the `finetuned_dino_detector` block (like the cuda_dino one in
`old_configs/`), feeding the resample+filter wiring in `notebooks/data_saving_evals/`.

---

## YOLO26 — same pattern, different model
**Step 1 (DONE + verified):** `yolo_training/export_yolo_models.sh` exports YOLO26 s+m to TorchScript
→ `yolo_training/weights/yolo26{s,m}.torchscript` (38.6 / 82.2 MB) + `.meta.json` (1 class "signal",
imgsz 1024). Container `model_script_path`:
`/workspace/holohub/yolo_training/weights/yolo26{s,m}.torchscript`.
- **Input:** `float[1,3,1024,1024]` in [0,1] (letterboxed RGB tile). **Output:** Ultralytics detection
  head raw predictions (decode + NMS in the operator).

**Step 2 (operator, needs container to compile):** `operators/yolo_detector/`. compute() front-end:
dB spectrogram (nfft=1024) → `db_to_uint8` (dataset_meta vmin/vmax) → 256-row tiles → replicate gray to
3ch → letterbox to 1024 → /255 → TorchScript → **decode + NMS** (conf 0.25, iou 0.45) → scale boxes
back to tile coords → **fill each box into the mask grid** (box→mask convention from
`yolo_training/src/yolo_infer.py`) → stitch → emit `DetectorMaskMessage`. Config block
`yolo_detector:` (`model_script_path`, `imgsz`, `conf`, `iou`, `tile_rows`, `nfft`, `db_vmin`,
`db_vmax`) + `DetectorAdapter` entries `yolo26s`/`yolo26m` + app CMake wiring. The decode/NMS/letterbox
is the trickiest part — marked `// TODO(lab-admin)` in the scaffold.

## Operator scaffolds (drafted)
`operators/finetuned_dino_detector/` and `operators/yolo_detector/` are scaffolded (CMakeLists.txt,
metadata.json, README.md, `.hpp` interface, `.cu` with setup/spec wired + a structured `compute()`
and `// TODO(lab-admin)` markers for the SDK-specific numeric bits). They are honest first drafts —
they will need an in-container compile pass to finish.

## Build + run everything at once
```bash
# export all weights (host):
conda activate dinov3 && dino_fine_tuning/export_finetuned_models.sh
conda activate yolo   && yolo_training/export_yolo_models.sh
# build all operators in one rebuild (lab-admin):
cd applications/usrp_wideband_signal_detection
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/build_all_detector_operators.sh
# run all 6 detectors offline over the FULL SNR sweep (default; skips any not yet wired):
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_all_detectors.sh
# ...or a quick single-SNR smoke test:
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_all_detectors.sh ~/captures/attenuation_dB_30.sigmf-data
```
`run_all_detectors.sh` runs coherent_power + cuda_dino today and auto-includes finetuned_dino /
finetuned_dino_m2 / yolo26s / yolo26m once each operator + its `config_<name>_single_channel.yaml`
exist. Masks land at `OUT_ROOT/<detector>/<stem>/mask_arrays/` → feed the snipper + the data-saving
notebook for the resample+filter numbers.

## Status (2026-07-21): operators fleshed out + wired (offline)
Both operators are now authored and **wired** — TorchScript `forward()` implemented (FT-DINO logits;
YOLO head decode + NMS), 4 configs created, 4 `DetectorAdapter` entries + a compose() raw-IQ branch
(`add_flow(source, detector)`), and app CMake `DEPENDS OPERATORS`/links done. Remaining = the
**CUDA/matx kernels only** (`TODO(lab-admin)`): nfft=1024 fftshift+power-dB, normalize/db_to_uint8,
tile packing, sigmoid-threshold (DINO) / letterbox+box-fill (YOLO), stitch, native→display-grid map,
device→host mask copy. `main.cpp` (live) is NOT wired yet — offline eval only.

**Build + validate:** `build_all_detector_operators.sh` (lab-admin) → `test_finetuned_dino_offline.sh`
(IoU vs Python `sweep_all` — the correctness gate). Top risks: raw-IQ port type, nfft=1024 ==
`rfdata.frames_to_db`, native→display map direction, YOLO head layout, torch stream ordering.

> **Heads-up:** the 4 configs now exist, so `run_all_detectors.sh` will **attempt** finetuned_dino /
> finetuned_dino_m2 / yolo26s / yolo26m (it no longer skips them) — they'll **error against the current
> (un-rebuilt) binary**. Rebuild the container first, then run them. coherent_power + cuda_dino are
> unaffected.

## Practical note
Operators/configs/exports are authored on the host, but **compiling/running requires the container
(docker/sudo = lab-admin)** — expect an author → build → fix → iterate loop for the first operator.
Once FT-DINO builds and its offline masks match the Python reference, YOLO26 is a mechanical repeat.

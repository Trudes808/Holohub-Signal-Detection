# Plan: `cuda_dino_finetuned` live Holoscan detector operator (REVISED)

Revised 2026-07-21. Supersedes the paused plan in `~/.claude/plans/jiggly-yawning-steele.md`.
Branch: `finetuned_dino_operator`.

## What changed vs the paused plan (and why)

1. **The paused "time-resolution mismatch" caveat is resolved.** The app README documents that the
   USRP X410 actually delivers **245.76 MSps even when `--rate 500e6` is requested** — the receiver
   derives FFT geometry from the real CHDR rate. That is exactly the fine-tuning capture rate. So the
   mismatch was never a sample-rate problem; it is purely an **FFT-window** problem: the app's analysis
   FFT uses a large nfft (~10240 → ~24 kHz/bin, ~41.7 µs/row) while the model wants **nfft=1024**
   (240 kHz/bin, 4.17 µs/row). Since the sample rate already matches, the operator computes its **own
   native 1024-pt / N-row FFT** and the physics match exactly — no unfixable time averaging.

2. **Front-end decision (locked): dedicated FFT in the operator.** The operator taps raw IQ and
   re-FFTs at the model's `nfft` (option (a) in bqn82's scaffold TODO), rather than downsampling the
   wide analysis FFT. This is the only route that matches training in *both* freq and time.

3. **The model input shape is CONFIG-DRIVEN, and the target shape is changing.** We are moving to a
   **512 time × 1024 freq** input (keep 240 kHz/bin, double the time context to ~2.13 ms/frame; token
   grid 32×64). The shipped M1/M2 checkpoints are 256×1024 and will NOT fit — a **retrain at 512×1024
   is required** and happens in parallel. The operator therefore never hard-codes the shape: `tile_rows`
   and `nfft` are config params, and `model_script_path` selects the checkpoint. Validate the operator
   first against the existing **256×1024 M2** checkpoint, then swap in the 512×1024 checkpoint once
   trained.

4. **Adopt bqn82's scaffold.** `bqn82`'s `operators/finetuned_dino_detector/` already establishes the
   Holoscan boundary correctly (same I/O contract as cuda_dino; params `model_script_path`, `threshold`,
   `tile_rows`, `nfft`, `db_vmin/vmax`, `num_channels`, `channel_filter`, `emit_stride`, `torch_dtype`;
   torch runtime split so nvcc never sees libtorch). We port it in and finish the `TODO(lab-admin)`
   front-end / `forward()` / post-proc.

## The core design: a single "geometry contract"

The DINOv3 ViT-B/16 + SegHead input grid is architecturally fixed **per checkpoint** (patch 16 →
token grid = tile_rows/16 × nfft/16; SegHead upsamples ×2 four times back to tile_rows × nfft). What
the fine-tune actually learns is tied to the **physical meaning per pixel**:

- freq: `bin_hz = sample_rate / nfft`  (training: 245.76e6 / 1024 = **240 kHz/bin**)
- time: `row_seconds = nfft / sample_rate`  (training: 1024 / 245.76e6 = **4.167 µs/row**)

Everything flexible flows from **one descriptor** shared by training and inference:

```
geometry = { sample_rate_hz, nfft, tile_rows, db_vmin, db_vmax, threshold }
         → derived: bin_hz = sample_rate_hz/nfft, row_seconds = nfft/sample_rate_hz
```

- **At export time** we emit this descriptor as a sidecar `*.meta.json` next to the `.ts`
  (the decision threshold + dB clip already need to travel with the model; this bundles them with the
  shape + physics).
- **At run time** the operator reads `tile_rows`, `nfft`, `db_vmin`, `db_vmax`, `threshold` from the
  config block (authoritative), and computes its dedicated FFT to that `nfft`. On load it MAY validate
  the config against the sidecar and warn on mismatch.
- **The only thing that differs** between "use our shipped model" and "retrain for a new receive
  setup" is this descriptor.

### Deployment decision tree (this is what the README's fine-tuning section teaches)

- **Same receive setup as training** (RX rate = 245.76 MSps): use the shipped checkpoint. Operator's
  dedicated FFT at nfft=1024 reproduces the trained physics exactly. No retrain.
- **Different RX rate, matched physics still reachable**: choose `nfft` so `bin_hz ≈ 240 kHz` and
  `row_seconds ≈ 4.17 µs`. If a single nfft satisfies both (i.e. the new rate is a near-integer
  multiple/divisor of 245.76 MSps), the shipped model still applies — set `nfft` in config. No retrain.
- **Different native bandwidth/resolution required, or physics can't be matched**: **retrain** at the
  deployment's native geometry. Edit one geometry block in `dino_fine_tuning/configs/dataset.yaml`,
  train, export → the new checkpoint's sidecar carries the new descriptor, and the operator adapts by
  setting `tile_rows`/`nfft`/`db_vmin`/`db_vmax`/`threshold` to match. Retrain.

## Phased implementation (user owns container build/run; Claude edits code + gives commands)

### Phase 0 — Port the scaffold  ✅ first
- Copy `operators/finetuned_dino_detector/{finetuned_dino_detector.hpp,.cu,
  finetuned_dino_torch_helpers.hpp,.cpp,CMakeLists.txt,metadata.json,README.md}` from bqn82's repo into
  this repo. Add `add_holohub_operator(finetuned_dino_detector)` to `operators/CMakeLists.txt`.

### Phase 1 — Export the fine-tuned segmenter to TorchScript (config-driven shape)
- New `export_dinov3_finetuned_torchscript.py` (modeled on `export_dinov3_torchscript.py`):
  args `--ckpt <best.pt> --tile-rows 256|512 --nfft 1024 --out <name>.ts`. Build `DinoSegmenter` via
  `finetuned_infer` load logic, `torch.jit.trace` on `randn(N,1,tile_rows,nfft)` fp32 under `no_grad`,
  `.save()`. Also **write `<name>.meta.json`** = the geometry contract (sample_rate, nfft, tile_rows,
  db_vmin, db_vmax, threshold, bin_hz, row_seconds).
- Verify: `.ts` vs eager model IoU ≈ 1 on real tiles.
- First export the **existing 256×1024 M2** for operator bring-up; the 512×1024 export follows the
  retrain.

### Phase 2 — Finish the operator (the `TODO(lab-admin)` bodies)
- **Front-end (dedicated FFT):** tap raw IQ (like `signal_snipper`) → matx FFT at `nfft` → fftshift →
  `power_db = 10·log10(|z|²+eps)` → `clamp((db-vmin)/(vmax-vmin),0,1)` → split rows into
  `ceil(rows/tile_rows)` tiles of `tile_rows` (pad last) → `[B,1,tile_rows,nfft]`. Register all new
  device buffers in the realloc path (per the [[coherent-detector-runtime-buffer-realloc]] lesson).
- **Inference:** finish `forward()` — `torch::from_blob` device tensor, CUDAStreamGuard, module forward,
  copy logits to device (reuse `dinov3_signal_detector/dinov3_torch_runtime.cpp` plumbing).
- **Post-proc:** `sigmoid(logits) >= threshold` kernel → stitch tiles → native rows×nfft uint8 mask →
  emit `DetectorMaskMessage` (device path, stream-synced).
- **IMPORTANT** the IQ tap changes the input contract vs the pure-FFT scaffold — decide the flow:
  IQ port from chdrConverter/spectrogram source into the operator; keep `mask_out` unchanged so snipper
  + viz gate wiring is reused.

### Phase 3 — Wire into the app
- `main.cpp`: `#include <finetuned_dino_detector.hpp>`; accept `detector_type == "cuda_dino_finetuned"`
  in the validation guard; add a make_operator branch (mirror cuda_dino incl. num_channels/
  channel_filter/emit_stride); on-screen label; add_operator; the three add_flow switches (IQ/source→op,
  op→snipper, op→viz gate). `applications/.../CMakeLists.txt`: link
  `holoscan::ops::finetuned_dino_detector` + include dir (+ offline-eval binary if desired).

### Phase 4 — Config
- New top-level `config_cuda_dino_finetuned_performance_single_channel.yaml`: clone cuda_dino config;
  `pipeline.detector_type: "cuda_dino_finetuned"`; `finetuned_dino_detector:` block with `tile_rows`,
  `nfft`, `db_vmin`, `db_vmax`, `threshold`, `model_script_path`, `torch_dtype`, `emit_stride` — and
  NONE of the fusion params. Ship it initially pointing at the **256×1024 M2** checkpoint for
  validation; document how to flip `tile_rows: 512` + the 512×1024 checkpoint path.

### Phase 5 — Build + verify (user runs)
- `sudo ./bash_scripts/rebuild_demo_container_app.sh` then
  `CONFIG_NAME=config_cuda_dino_finetuned_performance_single_channel.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh`.
- Confirm the mask overlay renders sensibly vs the offline finetuned masks. Optional offline
  apples-to-apples via `run_offline_cuda_detector_eval`.

### Phase 6 — README + docs (the user's headline ask)
- **High-level README quick-run**: the minimal command set to run the app with `cuda_dino_finetuned`
  using the shipped checkpoint, and where to put the `.ts`/`.meta.json`.
- **Fine-tuning section**:
  - (a) *For our receive setup* (245.76 MSps): how the shipped checkpoint was produced, checkpoint
    paths, reproduce commands (`dino_fine_tuning/scripts/run_full.sh`), export command.
  - (b) *For a different sample rate / spectrogram settings*: the decision tree above; the exact
    geometry knobs (`dataset.yaml` nfft/frame_rows/dB calibration), the retrain → export → config flow;
    and the config fields to change (`tile_rows`, `nfft`, `db_vmin`, `db_vmax`, `threshold`,
    `model_script_path`).
- **512×1024 retrain procedure** (for the parallel training effort): dataset.yaml `frame_rows: 512`,
  regenerate dataset, train, export at `--tile-rows 512`.
- Update CLAUDE.md experiment entry points + the path-scoped rule if it names the DINO config.

### Phase 7 — (deferred, only after verification) make finetuned the default
- Not part of initial delivery. Retire zero-shot cuda_dino config to `old_configs/` + repoint the
  runner default only after live verification and sign-off.

## Files
- CREATE: `operators/finetuned_dino_detector/*` (ported); `export_dinov3_finetuned_torchscript.py`;
  `config_cuda_dino_finetuned_performance_single_channel.yaml`.
- MODIFY: `operators/CMakeLists.txt`; `applications/usrp_wideband_signal_detection/{main.cpp,CMakeLists.txt}`;
  `README.md`; `CLAUDE.md`.

## Verification
1. Export `.ts` mask matches eager model (IoU≈1) at both 256×1024 (now) and 512×1024 (post-retrain).
2. Live run renders a sensible mask overlay; offline op-eval ≈ offline `dino_finetuned` mask.
3. Regression: existing `cuda_dino` / `coherent_power` configs build + run unchanged (additive branch).

## Open items to confirm during implementation
- IQ tap wiring: exact source operator/message for raw IQ into the detector (mirror `signal_snipper`).
- 512×1024 SegHead/positional interpolation: confirm DinoSegmenter trains cleanly at token grid 32×64.
- dB calibration (`db_vmin/vmax`) for the 512×1024 dataset may differ from the 256×1024 values.

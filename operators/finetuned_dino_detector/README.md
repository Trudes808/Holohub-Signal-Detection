<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation
SPDX-License-Identifier: Apache-2.0
-->
# Fine-tuned DINO Detector Operator (scaffold)

Native detector for the fine-tuned DINOv3 segmenter (`DinoSegmenter` = DINOv3 ViT-B/16 backbone +
trained SegHead), emitting `holoscan::ops::DetectorMaskMessage` on `mask_out` so `signal_snipper` can
snip it. Two model checkpoints: `finetuned_dino` (M1_ft, threshold 0.45) and `finetuned_dino_m2`
(M2_ft, threshold 0.85), selected via `model_script_path` + `threshold`.

## I/O contract
- **Input** `in`: `std::tuple<matx::tensor_t<complex<float>,2>, cudaStream_t>` (analysis FFT frame +
  stream), same as `cuda_dino_detector`.
- **TorchScript module** (`dino_fine_tuning/weights/finetuned_dino_m{1,2}.ts`): input
  `float[B,1,256,1024]` in [0,1] → output `logits[B,1,256,1024]`; post `sigmoid(logits) >= threshold`.
- **Output** `mask_out`: `DetectorMaskMessage` (native rows×nfft mask; `frame_number` + IQ offsets
  copied from the input frame metadata).

## Status
Scaffold — the Holoscan boundary (ports/params/receive/metadata/emit) mirrors `cuda_dino_detector`;
the numeric front-end (dB spectrogram at nfft=1024, tiling, normalize), the TorchScript `forward()`
device plumbing, and the sigmoid/threshold/stitch are marked `// TODO(lab-admin)`. See
`/workspace/holohub/build_instructions.md` and model on `operators/cuda_dino_detector/`.

<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation
SPDX-License-Identifier: Apache-2.0
-->
# YOLO26 Detector Operator (scaffold)

Native detector for the fine-tuned Ultralytics YOLO26 s/m detectors, emitting
`holoscan::ops::DetectorMaskMessage` on `mask_out` by filling predicted boxes into the mask grid
(box->mask, from `yolo_training/src/yolo_infer.py`). Two models: `yolo26s` / `yolo26m`, via
`model_script_path`.

## I/O contract
- **Input** `in`: `std::tuple<matx::tensor_t<complex<float>,2>, cudaStream_t>` (analysis FFT frame +
  stream), same as `cuda_dino_detector`.
- **TorchScript module** (`yolo_training/weights/yolo26{s,m}.torchscript`): input
  `float[1,3,imgsz,imgsz]` in [0,1] (letterboxed RGB tile) → Ultralytics detection head raw preds
  (single class "signal"); decode + NMS in the operator.
- **Output** `mask_out`: `DetectorMaskMessage` (native rows×nfft mask; `frame_number` + IQ offsets
  copied from the input frame metadata).

## Status
Scaffold — Holoscan boundary (ports/params/receive/metadata/emit) mirrors `cuda_dino_detector`. The
front-end (db→uint8, 256-row tiling, gray→3ch, letterbox to imgsz, /255), the TorchScript
`forward()`, the **decode + NMS** (`// TODO(lab-admin)` in `yolo_torch_helpers.cpp` — the trickiest
part; confirm the head output layout, e.g. [B,5,N]), and the box→mask fill + stitch are TODO. See
`/workspace/holohub/build_instructions.md` and model on `operators/cuda_dino_detector/`.

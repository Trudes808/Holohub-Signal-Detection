// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

// One decoded detection in letterboxed-image pixel coords (imgsz x imgsz).
struct YoloBox { float x0, y0, x1, y1, score; };

class YoloTorchRuntime {
 public:
  YoloTorchRuntime();
  ~YoloTorchRuntime();
  bool load(const std::string& model_script_path, const std::string& torch_dtype);
  bool loaded() const;

  // Forward one letterboxed batch (B x 3 x imgsz x imgsz float in [0,1], device) through the module,
  // decode the Ultralytics head, run NMS, and return kept boxes per image (letterboxed coords).
  // TODO(lab-admin): implement decode + NMS. Ultralytics single-class torchscript output is typically
  //   [B, 5, N] (channels = [cx,cy,w,h,score], N anchors) or the transposed [B, N, 5]; confirm the
  //   layout for this export. Steps: (per image) take preds, threshold score>=conf, convert cxcywh->
  //   xyxy (already in imgsz pixel units), NMS at `iou` (torchvision::nms or manual), emit YoloBox.
  bool forward(const float* letterbox_batch_device, int batch, int imgsz,
               float conf, float iou, std::vector<std::vector<YoloBox>>& boxes_per_image,
               cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops

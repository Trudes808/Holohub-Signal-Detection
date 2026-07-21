// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// TorchScript runtime for Ultralytics YOLO26. SCAFFOLD -- forward() decode+NMS is TODO(lab-admin).
#include "yolo_torch_helpers.hpp"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <vector>

#if defined(HOLOHUB_HAS_TORCH)
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace holoscan::ops {

struct YoloTorchRuntime::Impl {
#if defined(HOLOHUB_HAS_TORCH)
  torch::jit::script::Module module;
  bool is_loaded = false;
#endif
};

YoloTorchRuntime::YoloTorchRuntime() : impl_(std::make_unique<Impl>()) {}
YoloTorchRuntime::~YoloTorchRuntime() = default;

bool YoloTorchRuntime::load(const std::string& model_script_path, const std::string& torch_dtype) {
#if defined(HOLOHUB_HAS_TORCH)
  (void)torch_dtype;
  try {
    impl_->module = torch::jit::load(model_script_path, torch::kCUDA);
    impl_->module.eval();
    impl_->is_loaded = true;
    std::fprintf(stderr, "[yolo_detector] loaded TorchScript %s\n", model_script_path.c_str());
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[yolo_detector] ERROR loading %s: %s\n", model_script_path.c_str(), e.what());
    return false;
  }
#else
  (void)model_script_path; (void)torch_dtype; return false;
#endif
}

bool YoloTorchRuntime::loaded() const {
#if defined(HOLOHUB_HAS_TORCH)
  return impl_->is_loaded;
#else
  return false;
#endif
}

bool YoloTorchRuntime::forward(const float* letterbox_batch_device, int batch, int imgsz,
                               float conf, float iou,
                               std::vector<std::vector<YoloBox>>& boxes_per_image, cudaStream_t stream) {
#if defined(HOLOHUB_HAS_TORCH)
  if (!impl_->is_loaded) return false;
  boxes_per_image.assign(batch, {});
  try {
    torch::NoGradGuard no_grad;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto in = torch::from_blob(const_cast<float*>(letterbox_batch_device),
                               {batch, 3, imgsz, imgsz}, opts);
    torch::Tensor in_run = (impl_->dtype == torch::kHalf) ? in.to(torch::kHalf) : in;
    // Ultralytics detect head (v8+/26): output [B, 4+nc, N]; for nc=1 -> [B,5,N] with row 4 = class
    // prob and rows 0..3 = xywh in imgsz PIXELS. Some exports transpose to [B,N,5]. Decode on CPU.
    // TODO(lab-admin): confirm this export's exact layout (4+nc, coords-in-pixels, no separate
    // objectness) against `yolo predict` on one tile before trusting masks.
    auto out = impl_->module.forward({in_run}).toTensor().to(torch::kFloat32).contiguous().cpu();
    if (out.dim() != 3) {
      std::fprintf(stderr, "[yolo_detector] unexpected preds dim %ld\n", static_cast<long>(out.dim()));
      return false;
    }
    const int64_t d1 = out.size(1), d2 = out.size(2);
    const bool channels_first = (d1 == 5) || (d1 < d2);   // [B,5,N] vs [B,N,5]
    for (int b = 0; b < batch; ++b) {
      torch::Tensor p = out[b];
      if (channels_first) p = p.transpose(0, 1).contiguous();   // -> [N, 5]
      const int64_t N = p.size(0);
      if (p.size(1) < 5) { continue; }
      auto acc = p.accessor<float, 2>();
      std::vector<YoloBox> cand;
      cand.reserve(64);
      for (int64_t i = 0; i < N; ++i) {
        const float score = acc[i][4];
        if (score < conf) continue;
        const float cx = acc[i][0], cy = acc[i][1], w = acc[i][2], h = acc[i][3];
        cand.push_back(YoloBox{cx - 0.5f * w, cy - 0.5f * h, cx + 0.5f * w, cy + 0.5f * h, score});
      }
      std::sort(cand.begin(), cand.end(),
                [](const YoloBox& a, const YoloBox& c) { return a.score > c.score; });
      std::vector<char> removed(cand.size(), 0);
      std::vector<YoloBox> keep;
      for (size_t i = 0; i < cand.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(cand[i]);
        const float a1 = (cand[i].x1 - cand[i].x0) * (cand[i].y1 - cand[i].y0);
        for (size_t j = i + 1; j < cand.size(); ++j) {
          if (removed[j]) continue;
          const float xx0 = std::max(cand[i].x0, cand[j].x0), yy0 = std::max(cand[i].y0, cand[j].y0);
          const float xx1 = std::min(cand[i].x1, cand[j].x1), yy1 = std::min(cand[i].y1, cand[j].y1);
          const float iw = std::max(0.0f, xx1 - xx0), ih = std::max(0.0f, yy1 - yy0);
          const float inter = iw * ih;
          const float a2 = (cand[j].x1 - cand[j].x0) * (cand[j].y1 - cand[j].y0);
          const float uni = a1 + a2 - inter;
          if (uni > 0.0f && (inter / uni) > iou) removed[j] = 1;
        }
      }
      boxes_per_image[b] = std::move(keep);
    }
    (void)imgsz; (void)stream;
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[yolo_detector] forward error: %s\n", e.what());
    return false;
  }
#else
  (void)letterbox_batch_device; (void)batch; (void)imgsz; (void)conf; (void)iou; (void)stream;
  boxes_per_image.clear();
  return false;
#endif
}

}  // namespace holoscan::ops

// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// Native YOLO26 detector. SCAFFOLD: Holoscan boundary + emit mirror cuda_dino_detector; the YOLO
// front-end (db->uint8, tile, letterbox) and box->mask fill/stitch are marked TODO(lab-admin).
#include "yolo_detector.hpp"
#include "yolo_torch_helpers.hpp"
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <cuda_runtime.h>
#include <matx.h>

#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace holoscan::ops {

using yolo_complex = cuda::std::complex<float>;
using yolo_in_t = std::tuple<matx::tensor_t<yolo_complex, 2>, cudaStream_t>;

namespace {
inline void throw_if_cuda_error(cudaError_t err, const char* what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("[yolo_detector] ") + what + ": " + cudaGetErrorString(err));
}
}  // namespace

YoloDetector::~YoloDetector() { release_channel_buffers(); }

void YoloDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<yolo_in_t>("in");
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(model_script_path_, "model_script_path", "TorchScript path",
             "yolo26{s,m}.torchscript (container path).",
             std::string("/workspace/holohub/yolo_training/weights/yolo26m.torchscript"));
  spec.param(imgsz_, "imgsz", "Inference size", "Letterbox square size.", 1024);
  spec.param(conf_, "conf", "Confidence threshold", "Min box score.", 0.25);
  spec.param(iou_, "iou", "NMS IoU", "NMS IoU threshold.", 0.45);
  spec.param(tile_rows_, "tile_rows", "Tile rows", "Tile height.", 256);
  spec.param(nfft_, "nfft", "FFT size", "Native frequency bins.", 1024);
  spec.param(db_vmin_, "db_vmin", "dB vmin", "db_to_uint8 lower clip.", -100.0);
  spec.param(db_vmax_, "db_vmax", "dB vmax", "db_to_uint8 upper clip.", 0.0);
  spec.param(num_channels_, "num_channels", "Channels", "Detector channels.", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter", "Single-channel filter; <0 = all.", -1);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Process every Nth frame.", 1);
  spec.param(torch_dtype_, "torch_dtype", "Torch dtype", "fp32 | fp16.", std::string("fp32"));
}

void YoloDetector::initialize() {
  holoscan::Operator::initialize();
  const int channels = channel_filter_.get() >= 0 ? 1 : std::max(1, num_channels_.get());
  frame_count_.assign(channels, 0);
  channel_buffers_.assign(channels, ChannelBuffers{});
  runtime_ = std::make_shared<YoloTorchRuntime>();
  if (!runtime_->load(model_script_path_.get(), torch_dtype_.get()))
    std::fprintf(stderr, "[yolo_detector] WARN: TorchScript not loaded at init\n");
}

void YoloDetector::stop() { release_channel_buffers(); }

void YoloDetector::release_channel_buffers() {
  for (auto& b : channel_buffers_) {
    cudaFree(b.spectrogram_db_device);
    cudaFree(b.u8_device);
    cudaFree(b.letterbox_batch_device);
    cudaFree(b.stitched_mask_device);
    if (b.processing_stream) cudaStreamDestroy(b.processing_stream);
    b = ChannelBuffers{};
  }
}

void YoloDetector::compute(holoscan::InputContext& op_input,
                           holoscan::OutputContext& op_output,
                           holoscan::ExecutionContext&) {
  auto maybe_input = op_input.receive<yolo_in_t>("in");
  if (!maybe_input) return;
  const auto& [fft_tensor, fft_stream] = *maybe_input;

  auto meta = metadata();
  const uint16_t channel_number = meta ? meta->get<uint16_t>("channel_number", 0) : 0;
  const int channel_filter = channel_filter_.get();
  if (channel_filter >= 0 && channel_number != static_cast<uint16_t>(channel_filter)) return;
  const size_t ch = channel_filter >= 0 ? 0u : static_cast<size_t>(channel_number);
  if (ch >= channel_buffers_.size()) return;

  uint64_t frame_number = 0;
  if (meta && meta->has_key("fft_emitted_frame_number")) {
    frame_number = meta->get<uint64_t>("fft_emitted_frame_number", frame_count_[ch] + 1);
    if (frame_number == 0) frame_number = frame_count_[ch] + 1;
    frame_count_[ch] = std::max(frame_count_[ch], frame_number);
  } else {
    frame_number = ++frame_count_[ch];
  }
  if (meta && meta->get<bool>("offline_source_drain_frame", false)) return;
  if (emit_stride_.get() > 1 && (frame_number % emit_stride_.get()) != 0) return;

  const int nfft = nfft_.get();          // 1024
  const int tile_rows = tile_rows_.get();  // 256
  const int imgsz = imgsz_.get();          // 1024

  // ---- geometry (raw-IQ path, identical to finetuned_dino_detector) --------------------------
  // fft_tensor is RAW IQ (num_bursts x burst_size) = (512 x 10240); re-FFT at nfft=1024 to match
  // yolo_training/src/yolo_infer.py (frames_to_db + db_to_uint8).
  const int num_bursts = static_cast<int>(fft_tensor.Size(0));   // 512  (display/emit time rows)
  const int burst_size = static_cast<int>(fft_tensor.Size(1));   // 10240 (display/emit freq cols)
  const long total_samples = static_cast<long>(num_bursts) * static_cast<long>(burst_size);
  const int model_rows = static_cast<int>(total_samples / nfft); // 5120
  const int batch = (model_rows + tile_rows - 1) / tile_rows;    // 20 tiles
  auto& buf = channel_buffers_[ch];
  if (buf.processing_stream == nullptr)
    throw_if_cuda_error(cudaStreamCreate(&buf.processing_stream), "stream create");

  // ---- front-end (TODO lab-admin CUDA/matx) : yolo_infer.mask_for_iq -------------------------
  //   1. reshape raw IQ (total_samples) -> (model_rows, nfft); spec = fftshift(fft(row)); power_db.
  //   2. u8 = db_to_uint8(power_db, db_vmin, db_vmax)  = clip((db-vmin)/(vmax-vmin),0,1)*255  (uint8)
  //   3. split into ceil(model_rows/tile_rows) tiles of tile_rows (pad last with 0)
  //   4. per tile: replicate gray->3ch, LETTERBOX 256x1024 -> imgsz x imgsz (record scale s + pad
  //      (px,py) so boxes map back: x_tile=(x_lb-px)/s, y_tile=(y_lb-py)/s), then /255 ->
  //      buf.letterbox_batch_device [B,3,imgsz,imgsz]. (Ultralytics letterbox: s=min(imgsz/tile_rows,
  //      imgsz/nfft), centered padding.)

  std::vector<std::vector<YoloBox>> boxes_per_tile;   // decoded+NMS'd boxes in letterboxed px, per tile
  bool ok = runtime_ && runtime_->loaded() &&
            runtime_->forward(buf.letterbox_batch_device, batch, imgsz,
                              static_cast<float>(conf_.get()), static_cast<float>(iou_.get()),
                              boxes_per_tile, buf.processing_stream);
  if (!ok) {
    if (!startup_log_emitted_) {
      std::fprintf(stderr, "[yolo_detector] inference unavailable (front-end/forward TODO)\n");
      startup_log_emitted_ = true;
    }
    return;
  }

  // ---- post-proc (TODO lab-admin CUDA) : box->mask fill + stitch + display grid --------------
  //   5. un-letterbox each YoloBox -> tile pixel coords (undo step-4 s + pad); clamp to [0,tile_rows]/[0,nfft]
  //   6. per tile, fill native mask[y0:y1, x0:x1]=1 for every box; write at the tile's row offset
  //      (drop padded rows on the last tile) -> buf.stitched_mask_device (model_rows x nfft)
  //   7. map native (model_rows x nfft) -> DISPLAY grid (num_bursts x burst_size) [MAX-pool rows,
  //      NEAREST-up cols], same as finetuned_dino_detector, so masks align with the sweep_all reference.
  const int input_rows = num_bursts;    // display grid (512)
  const int input_cols = burst_size;    // 10240

  holoscan::ops::DetectorMaskMessage mask_msg;
  mask_msg.width = input_cols;
  mask_msg.height = input_rows;
  mask_msg.channel = channel_number;
  mask_msg.frame_number = frame_number;
  // TODO(lab-admin): fill display-grid mask into mask_msg.pixels (resize num_bursts*burst_size,
  //   cudaMemcpy device->host after synchronizing buf.processing_stream).
  if (meta) {
    mask_msg.file_offset_complex   = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
    mask_msg.data_end_complex      = meta->get<uint64_t>("offline_source_data_end_complex", 0);
    mask_msg.frame_end_complex     = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
    mask_msg.complex_samples_read  = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
    mask_msg.complex_samples_padded= meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
  }
  op_output.emit(mask_msg, "mask_out");
  if (meta) meta->set("yolo_mask_emitted", true);
  ++compute_count_;
}

}  // namespace holoscan::ops

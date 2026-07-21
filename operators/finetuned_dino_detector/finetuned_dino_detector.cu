// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// Native fine-tuned DINOv3 segmenter detector. SCAFFOLD: the Holoscan boundary (ports, params,
// input receive, metadata/frame bookkeeping, DetectorMaskMessage emit) mirrors cuda_dino_detector
// and is intended to be correct; the numeric front-end/post-proc CUDA is marked TODO(lab-admin).
#include "finetuned_dino_detector.hpp"
#include "finetuned_dino_torch_helpers.hpp"
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <cuda_runtime.h>
#include <matx.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <tuple>

namespace holoscan::ops {

using ft_dino_complex = cuda::std::complex<float>;
// Same input contract as cuda_dino_detector: the analysis FFT frame (rows x freq) + its stream.
using ft_dino_in_t = std::tuple<matx::tensor_t<ft_dino_complex, 2>, cudaStream_t>;

namespace {
inline void throw_if_cuda_error(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("[finetuned_dino_detector] ") + what + ": " +
                             cudaGetErrorString(err));
  }
}
}  // namespace

FinetunedDinoDetector::~FinetunedDinoDetector() { release_channel_buffers(); }

void FinetunedDinoDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<ft_dino_in_t>("in");
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(model_script_path_, "model_script_path", "TorchScript path",
             "Path to finetuned_dino_m{1,2}.ts (container path).",
             std::string("/workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m1.ts"));
  spec.param(threshold_, "threshold", "Decision threshold", "sigmoid(logits) >= threshold.", 0.45);
  spec.param(tile_rows_, "tile_rows", "Tile rows", "Model tile height.", 256);
  spec.param(nfft_, "nfft", "FFT size", "Model native frequency bins.", 1024);
  spec.param(db_vmin_, "db_vmin", "dB vmin", "Lower dB clip for [0,1] normalization.", -100.0);
  spec.param(db_vmax_, "db_vmax", "dB vmax", "Upper dB clip for [0,1] normalization.", 0.0);
  spec.param(num_channels_, "num_channels", "Channels", "Detector channels.", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter", "Single-channel filter; <0 = all.", -1);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Process every Nth frame.", 1);
  spec.param(torch_dtype_, "torch_dtype", "Torch dtype", "fp32 | fp16.", std::string("fp32"));
}

void FinetunedDinoDetector::initialize() {
  holoscan::Operator::initialize();
  const int channels = channel_filter_.get() >= 0 ? 1 : std::max(1, num_channels_.get());
  frame_count_.assign(channels, 0);
  channel_buffers_.assign(channels, ChannelBuffers{});
  runtime_ = std::make_shared<FinetunedDinoTorchRuntime>();
  if (!runtime_->load(model_script_path_.get(), torch_dtype_.get())) {
    std::fprintf(stderr, "[finetuned_dino_detector] WARN: TorchScript not loaded at init\n");
  }
}

void FinetunedDinoDetector::stop() { release_channel_buffers(); }

void FinetunedDinoDetector::release_channel_buffers() {
  for (auto& b : channel_buffers_) {
    cudaFree(b.spectrogram_db_device);
    cudaFree(b.normalized_device);
    cudaFree(b.tile_batch_device);
    cudaFree(b.logits_device);
    cudaFree(b.tile_mask_device);
    cudaFree(b.stitched_mask_device);
    if (b.processing_stream) cudaStreamDestroy(b.processing_stream);
    b = ChannelBuffers{};
  }
}

void FinetunedDinoDetector::compute(holoscan::InputContext& op_input,
                                    holoscan::OutputContext& op_output,
                                    holoscan::ExecutionContext&) {
  auto maybe_input = op_input.receive<ft_dino_in_t>("in");
  if (!maybe_input) return;
  const auto& [fft_tensor, fft_stream] = *maybe_input;

  auto meta = metadata();
  const uint16_t channel_number = meta ? meta->get<uint16_t>("channel_number", 0) : 0;
  const int channel_filter = channel_filter_.get();
  if (channel_filter >= 0 && channel_number != static_cast<uint16_t>(channel_filter)) return;
  const size_t ch = channel_filter >= 0 ? 0u : static_cast<size_t>(channel_number);
  if (ch >= channel_buffers_.size()) return;

  // frame_number: prefer the FFT-stamped counter so masks align to IQ frames (matches cuda_dino).
  uint64_t frame_number = 0;
  if (meta && meta->has_key("fft_emitted_frame_number")) {
    frame_number = meta->get<uint64_t>("fft_emitted_frame_number", frame_count_[ch] + 1);
    if (frame_number == 0) frame_number = frame_count_[ch] + 1;
    frame_count_[ch] = std::max(frame_count_[ch], frame_number);
  } else {
    frame_number = ++frame_count_[ch];
  }
  // Offline drain / partial-batch / emit-stride frames: skip without emitting (see cuda_dino_detector).
  if (meta && meta->get<bool>("offline_source_drain_frame", false)) return;
  if (emit_stride_.get() > 1 && (frame_number % emit_stride_.get()) != 0) return;

  const int nfft = nfft_.get();          // 1024 (model native)
  const int tile_rows = tile_rows_.get();  // 256

  // ---- geometry ------------------------------------------------------------------------------
  // With the raw-IQ compose path (add_flow(source, detector)), `fft_tensor` is the RAW IQ frame:
  // (num_bursts x burst_size) complex = (512 x 10240) here. We re-FFT it at nfft=1024 to match the
  // model's training geometry (dino_fine_tuning/src/finetuned_infer.mask_for_iq): flatten to a
  // contiguous IQ stream, reshape (model_rows x nfft), FFT along freq with fftshift, power-dB.
  const int num_bursts = static_cast<int>(fft_tensor.Size(0));   // 512  (display/emit time rows)
  const int burst_size = static_cast<int>(fft_tensor.Size(1));   // 10240 (display/emit freq cols)
  const long total_samples = static_cast<long>(num_bursts) * static_cast<long>(burst_size);
  const int model_rows = static_cast<int>(total_samples / nfft); // 5120 for nfft=1024
  const int batch = (model_rows + tile_rows - 1) / tile_rows;    // 20 tiles of 256 rows
  auto& buf = channel_buffers_[ch];
  if (buf.processing_stream == nullptr) {
    throw_if_cuda_error(cudaStreamCreate(&buf.processing_stream), "stream create");
  }

  // ---- front-end (TODO lab-admin CUDA/matx) --------------------------------------------------
  //   1. view fft_tensor's device data as total_samples contiguous complex; reshape (model_rows,nfft)
  //   2. spec = fftshift(fft(row), dim=freq)   (matx: matx::fft over last dim + fftshift) -- match
  //      rfdata.frames_to_db exactly (per-row fft + fftshift, DC centered).
  //   3. power_db = 10*log10(re^2+im^2+1e-12)                        -> buf.spectrogram_db_device
  //   4. normalized = clamp((power_db - db_vmin)/(db_vmax - db_vmin), 0, 1)  -> buf.normalized_device
  //   5. pack into [B,1,tile_rows,nfft] tiles (pad last tile rows with 0)    -> buf.tile_batch_device
  // (re)allocate buf.* for {model_rows,nfft,batch,num_bursts,burst_size} when they change.

  // ---- inference (real: torch runtime) -------------------------------------------------------
  bool ok = runtime_ && runtime_->loaded() &&
            runtime_->forward(buf.tile_batch_device, batch, tile_rows, nfft,
                              buf.logits_device, buf.processing_stream);
  if (!ok) {
    if (!startup_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] inference unavailable (front-end/forward TODO)\n");
      startup_log_emitted_ = true;
    }
    return;  // don't emit a bogus mask
  }

  // ---- post-proc (TODO lab-admin CUDA) -------------------------------------------------------
  //   6. tile_mask = (sigmoid(logits) >= threshold)                 (kernel over B*tile_rows*nfft)
  //   7. stitch tiles -> native (model_rows x nfft) binary mask      -> buf.stitched_mask_device
  //   8. map native (model_rows x nfft) -> DISPLAY grid (num_bursts x burst_size), matching
  //      finetuned_infer.to_display_grid: rows shrink 5120->512 = MAX-pool (block of model_rows/num_bursts);
  //      cols grow 1024->10240 = NEAREST up (out_col c -> native col c*nfft/burst_size). This makes the
  //      emitted mask align with the sweep_all reference (512 x 10240) the notebook + IoU check use.
  const int input_rows = num_bursts;    // emit at the display grid (512), NOT the native 5120
  const int input_cols = burst_size;    // 10240

  holoscan::ops::DetectorMaskMessage mask_msg;
  mask_msg.width = input_cols;
  mask_msg.height = input_rows;
  mask_msg.channel = channel_number;
  mask_msg.frame_number = frame_number;
  // TODO(lab-admin): fill the display-grid mask. Simplest robust path (mask_arrays sink reads host
  // `pixels`): mask_msg.pixels.resize((size_t)input_rows*input_cols); cudaMemcpy the display-grid
  // uint8 mask device->host into it (synchronize buf.processing_stream first). (device_pixels is the
  // faster alternative, like cuda_dino, if the sink supports it.)
  if (meta) {
    mask_msg.file_offset_complex   = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
    mask_msg.data_end_complex      = meta->get<uint64_t>("offline_source_data_end_complex", 0);
    mask_msg.frame_end_complex     = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
    mask_msg.complex_samples_read  = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
    mask_msg.complex_samples_padded= meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
  }
  op_output.emit(mask_msg, "mask_out");
  if (meta) meta->set("finetuned_dino_mask_emitted", true);
  ++compute_count_;
}

}  // namespace holoscan::ops

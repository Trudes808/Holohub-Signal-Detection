// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// Native fine-tuned DINOv3 segmenter detector. Taps raw time-domain IQ (like signal_snipper),
// computes a dedicated nfft-point FFT -> dB spectrogram at the model's trained geometry, normalizes
// to [0,1], tiles into tile_rows-row inputs, runs the TorchScript segmenter, thresholds sigmoid
// logits, stitches the per-tile masks, and emits a native (rows x nfft) DetectorMaskMessage.
//
// Front-end math mirrors dino_fine_tuning/src/rfdata.frames_to_db + finetuned_infer.mask_for_iq:
//   spec = fftshift(fft(row), dim=freq)   (no window, no FFT normalization)
//   db   = 10*log10(|spec|^2 + 1e-12)
//   img  = clamp((db - db_vmin) / (db_vmax - db_vmin), 0, 1)
#include "finetuned_dino_detector.hpp"
#include "finetuned_dino_torch_helpers.hpp"
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <cuda_runtime.h>
#include <matx.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <tuple>

namespace holoscan::ops {

using ft_dino_complex = cuda::std::complex<float>;
// Same message type signal_snipper taps for raw IQ: (time-domain IQ frame, stream).
using ft_dino_iq_t = std::tuple<matx::tensor_t<ft_dino_complex, 2>, cudaStream_t>;

namespace {

inline void throw_if_cuda_error(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("[finetuned_dino_detector] ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

// Port of resolve_fft_runtime_config (fft_runtime_config.hpp): the deployed dynamic FFT size for a
// sample rate (power-of-two span snap around a 500 MHz / 20480-bin reference, quantized to 1024-sample
// packets). Used so the downsample wide FFT tracks the rate -> ~512 rows/frame -> bounded 2-tile cost
// at ANY rate (rather than a fixed size whose row count, hence compute, would scale with rate).
inline int auto_fft_size(double sample_rate_hz, double ref_span_hz = 500.0e6,
                         int ref_fft = 20480, int packet = 1024) {
  const double span_ratio = sample_rate_hz / ref_span_hz;
  const double snapped = (std::isfinite(span_ratio) && span_ratio > 0.0)
                             ? std::pow(2.0, std::round(std::log2(span_ratio))) : 1.0;
  const int packets = std::max(1, static_cast<int>(std::round(ref_fft * snapped / packet)));
  return std::max(packet, packets * packet);
}

// spec (row-major rows x nfft complex, unshifted FFT output) -> normalized [0,1] float with an
// in-place fftshift along the frequency axis (display col c reads raw col (c + nfft/2) % nfft).
__global__ void power_db_normalize_kernel(const float2* __restrict__ spec,
                                          float* __restrict__ out,
                                          int rows, int nfft,
                                          float vmin, float inv_span) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long total = (long)rows * nfft;
  if (idx >= total) return;
  const int col = idx % nfft;
  const int row = idx / nfft;
  const int src_col = (col + nfft / 2) % nfft;
  const float2 c = spec[(long)row * nfft + src_col];
  const float power = c.x * c.x + c.y * c.y + 1e-12f;
  const float db = 10.0f * log10f(power);
  float v = (db - vmin) * inv_span;
  v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
  out[idx] = v;
}

// ---- optional per-frequency noise-floor flatten (adapted from coherent_power's frontend) ----------
// The segmenter is trained on flat-floor spectrograms; a receiver's analog/digital filter shapes the
// live noise floor (rolloff/tilt at the band edges), which the model can fire on. We estimate a smooth
// per-frequency floor from the frame and additively LIFT low-floor bins up to a data-derived reference
// (capped), flattening the floor without pulling real signals down. Coherent_power does this with
// freq on the ROW axis; here freq is the COLUMN axis, and it runs on the dB image before the [0,1]
// clip. Fully dynamic (per-frame) so it reproduces on any OTA capture with no calibration file.

// spec -> dB with fftshift along freq + FFT processing-gain correction (subtracted), no clip.
__global__ void ft_power_to_db_shift_kernel(const float2* __restrict__ spec, float* __restrict__ db_out,
                                            int rows, int nfft, float gain_offset_db) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long total = (long)rows * nfft;
  if (idx >= total) return;
  const int col = idx % nfft;
  const int row = idx / nfft;
  const int src_col = (col + nfft / 2) % nfft;
  const float2 c = spec[(long)row * nfft + src_col];
  const float power = c.x * c.x + c.y * c.y + 1e-12f;
  db_out[idx] = 10.0f * log10f(power) - gain_offset_db;
}

// Per-frequency (column) mean of dB over the time rows, optionally capped at reference+headroom so a
// strong signal in a bin does not inflate that bin's floor estimate. reference==nullptr => no cap.
__global__ void ft_col_mean_kernel(const float* __restrict__ db, int rows, int nfft,
                                   const float* reference, float cap_headroom_db,
                                   float* __restrict__ col_stat) {
  const int col = blockIdx.x;
  if (col >= nfft) return;
  __shared__ float partial[256];
  const int tid = threadIdx.x;
  const float cap = reference ? (reference[0] + fmaxf(cap_headroom_db, 0.0f)) : 3.0e38f;
  float sum = 0.0f;
  for (int row = tid; row < rows; row += blockDim.x) {
    sum += fminf(db[(long)row * nfft + col], cap);
  }
  partial[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) partial[tid] += partial[tid + s]; __syncthreads(); }
  if (tid == 0) col_stat[col] = partial[0] / (float)max(rows, 1);
}

// Gaussian smooth the per-frequency floor along frequency (columns).
__global__ void ft_smooth_cols_kernel(const float* __restrict__ in, int nfft, int radius, float sigma,
                                      float* __restrict__ out) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= nfft) return;
  float sum = 0.0f, wsum = 0.0f;
  for (int off = -radius; off <= radius; ++off) {
    const int s = max(0, min(nfft - 1, col + off));
    const float w = expf(-(float)(off * off) / (2.0f * sigma * sigma));
    sum += in[s] * w; wsum += w;
  }
  out[col] = wsum > 0.0f ? sum / wsum : in[col];
}

// Reference floor level = blend(mean -> max) of the smoothed per-freq floor (quantile in [0.5,1]).
__global__ void ft_frontend_reference_kernel(const float* __restrict__ col_smooth, int nfft,
                                             float quantile, float* __restrict__ reference) {
  __shared__ float psum[256];
  __shared__ float pmax[256];
  const int tid = threadIdx.x;
  float s = 0.0f, m = -3.0e38f;
  for (int c = tid; c < nfft; c += blockDim.x) { const float v = col_smooth[c]; s += v; m = fmaxf(m, v); }
  psum[tid] = s; pmax[tid] = m;
  __syncthreads();
  for (int st = blockDim.x / 2; st > 0; st >>= 1) {
    if (tid < st) { psum[tid] += psum[tid + st]; pmax[tid] = fmaxf(pmax[tid], pmax[tid + st]); }
    __syncthreads();
  }
  if (tid == 0) {
    const float mean = psum[0] / (float)max(nfft, 1);
    const float blend = fminf(fmaxf((quantile - 0.5f) / 0.5f, 0.0f), 1.0f);
    reference[0] = mean + blend * (pmax[0] - mean);
  }
}

// Lift each column in place by clamp(reference - smooth_floor[col], 0, max_boost) -> flat floor.
__global__ void ft_frontend_correction_kernel(float* __restrict__ db, int rows, int nfft,
                                              const float* __restrict__ col_smooth,
                                              const float* __restrict__ reference, float max_boost_db) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long total = (long)rows * nfft;
  if (idx >= total) return;
  const int col = idx % nfft;
  const float boost = fminf(fmaxf(reference[0] - col_smooth[col], 0.0f), max_boost_db);
  db[idx] += boost;
}

// dB -> [0,1] clip (flatten path; fftshift already applied in ft_power_to_db_shift_kernel).
__global__ void ft_clip_normalize_kernel(const float* __restrict__ db, float* __restrict__ out,
                                         long total, float vmin, float inv_span) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float v = (db[idx] - vmin) * inv_span;
  out[idx] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// sigmoid(logit) >= threshold -> uint8 {0,1}
__global__ void sigmoid_threshold_kernel(const float* __restrict__ logits,
                                         uint8_t* __restrict__ mask,
                                         long n, float threshold) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float p = 1.0f / (1.0f + expf(-logits[idx]));
  mask[idx] = (p >= threshold) ? 1 : 0;
}

}  // namespace

void FinetunedDinoDetector::ChannelBuffers::release() {
  cudaFree(spec_device);
  cudaFree(normalized_device);
  cudaFree(db_device);
  cudaFree(col_stat_device);
  cudaFree(col_smooth_device);
  cudaFree(frontend_reference_device);
  cudaFree(tile_batch_device);
  cudaFree(logits_device);
  cudaFree(tile_mask_device);
  cudaFree(window_device);
  spec_device = nullptr;
  normalized_device = nullptr;
  db_device = nullptr;
  col_stat_device = nullptr;
  col_smooth_device = nullptr;
  frontend_reference_device = nullptr;
  tile_batch_device = nullptr;
  logits_device = nullptr;
  tile_mask_device = nullptr;
  window_device = nullptr;
  window_len = 0;
  rows = nfft = tile_rows = batch = 0;
}

void FinetunedDinoDetector::ChannelBuffers::ensure(size_t new_rows, size_t new_nfft,
                                                   size_t new_tile_rows, size_t new_batch) {
  if (new_rows == rows && new_nfft == nfft && new_tile_rows == tile_rows && new_batch == batch) {
    return;
  }
  // Free the stale buffers and reallocate for the new geometry (see the coherent-detector
  // runtime-realloc lesson: EVERY device buffer must be re-registered here or it OOBs).
  cudaFree(spec_device);
  cudaFree(normalized_device);
  cudaFree(db_device);
  cudaFree(col_stat_device);
  cudaFree(col_smooth_device);
  cudaFree(frontend_reference_device);
  cudaFree(tile_batch_device);
  cudaFree(logits_device);
  cudaFree(tile_mask_device);
  cudaFree(window_device);

  const size_t spec_elems = new_rows * new_nfft;
  const size_t tile_elems = new_batch * new_tile_rows * new_nfft;
  throw_if_cuda_error(cudaMalloc(&window_device, new_nfft * sizeof(float)), "malloc window");
  window_len = 0;  // refilled in compute() for the current window type + fft_size
  throw_if_cuda_error(cudaMalloc(&spec_device, spec_elems * sizeof(ft_dino_complex)), "malloc spec");
  throw_if_cuda_error(cudaMalloc(&normalized_device, spec_elems * sizeof(float)), "malloc normalized");
  // Flatten scratch: full dB image + per-frequency (nfft-wide) floor buffers + scalar reference.
  throw_if_cuda_error(cudaMalloc(&db_device, spec_elems * sizeof(float)), "malloc db");
  throw_if_cuda_error(cudaMalloc(&col_stat_device, new_nfft * sizeof(float)), "malloc col_stat");
  throw_if_cuda_error(cudaMalloc(&col_smooth_device, new_nfft * sizeof(float)), "malloc col_smooth");
  throw_if_cuda_error(cudaMalloc(&frontend_reference_device, sizeof(float)), "malloc frontend_reference");
  throw_if_cuda_error(cudaMalloc(&tile_batch_device, tile_elems * sizeof(float)), "malloc tiles");
  throw_if_cuda_error(cudaMalloc(&logits_device, tile_elems * sizeof(float)), "malloc logits");
  throw_if_cuda_error(cudaMalloc(&tile_mask_device, tile_elems * sizeof(uint8_t)), "malloc tile_mask");
  rows = new_rows;
  nfft = new_nfft;
  tile_rows = new_tile_rows;
  batch = new_batch;
}

FinetunedDinoDetector::~FinetunedDinoDetector() { release_channel_buffers(); }

void FinetunedDinoDetector::setup(holoscan::OperatorSpec& spec) {
  // Raw time-domain IQ tapped upstream of the FFT (same stream/type signal_snipper consumes).
  spec.input<ft_dino_iq_t>("iq_in");
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(model_script_path_, "model_script_path", "TorchScript path",
             "Path to the exported fine-tuned segmenter .ts (container path).",
             std::string("/workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m2.ts"));
  spec.param(fft_window_, "fft_window", "FFT window",
             "Analysis window applied along the freq axis before the FFT to suppress spectral leakage: "
             "hann|hamming|blackman|none. MUST match the window the model was trained with (in the "
             "checkpoint's meta.json / dataset fft_window).", std::string("hann"));
  spec.param(threshold_, "threshold", "Decision threshold", "sigmoid(logits) >= threshold.", 0.85);
  spec.param(tile_rows_, "tile_rows", "Tile rows", "Model input time rows (mult of 16).", 256);
  spec.param(nfft_, "nfft", "FFT size", "Model input frequency bins (mult of 16).", 1024);
  spec.param(db_vmin_, "db_vmin", "dB vmin", "Lower dB clip for [0,1] normalization.", -46.934);
  spec.param(db_vmax_, "db_vmax", "dB vmax", "Upper dB clip for [0,1] normalization.", 19.557);
  spec.param(num_channels_, "num_channels", "Channels", "Detector channels.", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter", "Single-channel filter; <0 = all.", -1);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Process every Nth frame.", 1);
  spec.param(torch_dtype_, "torch_dtype", "Torch dtype", "fp32 | fp16.", std::string("fp32"));
  spec.param(real_time_downsample_, "real_time_downsample", "Downsample mode",
             "Wide FFT + bilinear resize freq to model width before inference (bounded cost). "
             "false = native per-tile path.", false);
  spec.param(downsample_fft_size_, "downsample_fft_size", "Downsample FFT size",
             "Wide FFT size for downsample mode (reproduces the app's dynamic spectrogram; e.g. 10240 "
             "at 245.76 MSps, 20480 at 500 MHz). Ignored when real_time_downsample=false.", 10240);
  spec.param(match_training_power_level_, "match_training_power_level", "Match training power level",
             "OPT-IN rate/RBW term: add 10*log10(rate/reference_sample_rate_hz) to the vmin shift. Only "
             "correct when the deployment GAIN matches the training captures; otherwise it can be the "
             "wrong sign (use power_level_trim_db instead). Default off.", false);
  spec.param(reference_sample_rate_hz_, "reference_sample_rate_hz", "Reference sample rate",
             "Sample rate the checkpoint was trained at (its captures were 245.76 MS/s). Used only for "
             "the power-level match.", 245760000.0);
  spec.param(power_level_trim_db_, "power_level_trim_db", "Power level trim",
             "Manual scalar dB nudge added to the level match to absorb gain/antenna/cable differences "
             "between the training captures and deployment. 0 = pure rate-derived.", 0.0);
  // Default OFF: the training preprocessing (dino_fine_tuning frames_to_db) does NO flattening -- the
  // model was trained on raw dB spectrograms that CONTAIN the receiver envelope, so flattening imposes
  // a shape the model never saw. Kept as an opt-in tool for models trained on flat-floor data.
  spec.param(flatten_noise_floor_, "flatten_noise_floor", "Flatten noise floor",
             "Estimate a smooth per-frequency floor and lift low-floor bins up to a data-derived "
             "reference before inference. OFF by default: this model was trained WITH the envelope "
             "present, so flattening is out-of-distribution.", false);
  spec.param(flatten_reference_q_, "flatten_reference_q", "Flatten reference quantile",
             "Blend mean->max of the per-freq floor for the reference level (50-100). Dimensionless "
             "-> bandwidth-invariant.", 75.0);
  spec.param(flatten_smooth_frac_, "flatten_smooth_frac", "Flatten smoothing fraction",
             "Gaussian sigma for the floor estimate as a FRACTION of the FFT size, so the smoothing "
             "spans a fixed fraction of the band at any sample rate (the filter rolloff is a fixed "
             "fraction of Nyquist). Wide enough to skip narrowband signals, narrow enough to follow "
             "the rolloff. sigma = max(2, frac*fft_size).", 0.005);
  spec.param(flatten_max_boost_db_, "flatten_max_boost_db", "Flatten max boost",
             "Max additive lift per frequency bin (dB).", 12.0);
  spec.param(flatten_signal_cap_db_, "flatten_signal_cap_db", "Flatten signal cap",
             "Cap a bin's influence on its own floor estimate at reference+this (dB) so strong "
             "signals don't inflate the floor; 0 disables the capped second pass.", 6.0);
}

void FinetunedDinoDetector::initialize() {
  holoscan::Operator::initialize();
  const int channels = channel_filter_.get() >= 0 ? 1 : std::max(1, num_channels_.get());
  frame_count_.assign(channels, 0);
  channel_buffers_.assign(channels, ChannelBuffers{});
  runtime_ = std::make_shared<FinetunedDinoTorchRuntime>();
  if (!runtime_->load(model_script_path_.get(), torch_dtype_.get())) {
    std::fprintf(stderr, "[finetuned_dino_detector] WARN: TorchScript not loaded at init\n");
  } else {
    // Prime the model so the first real frame doesn't pay the ~hundreds-of-ms cuDNN-autotune cost
    // (which would otherwise spike backpressure at live startup). Tiles are tile_rows x nfft; batch 2
    // covers the downsample path (512 rows) and primes the same per-tile kernels the native path uses.
    runtime_->warmup(tile_rows_.get(), nfft_.get(), 2);
  }
}

void FinetunedDinoDetector::stop() { release_channel_buffers(); }

void FinetunedDinoDetector::release_channel_buffers() {
  for (auto& b : channel_buffers_) {
    b.release();
  }
}

void FinetunedDinoDetector::compute(holoscan::InputContext& op_input,
                                    holoscan::OutputContext& op_output,
                                    holoscan::ExecutionContext&) {
  auto maybe_input = op_input.receive<ft_dino_iq_t>("iq_in");
  if (!maybe_input) return;
  auto iq_tensor = std::get<0>(maybe_input.value());  // matx tensor_t is a shallow handle
  const cudaStream_t iq_stream = std::get<1>(maybe_input.value());

  auto meta = metadata();
  const uint16_t channel_number = meta ? meta->get<uint16_t>("channel_number", 0) : 0;
  const int channel_filter = channel_filter_.get();
  if (channel_filter >= 0 && channel_number != static_cast<uint16_t>(channel_filter)) return;
  const size_t ch = channel_filter >= 0 ? 0u : static_cast<size_t>(channel_number);
  if (ch >= channel_buffers_.size()) return;

  // Offline drain / partial-batch frames: skip without emitting (matches cuda_dino_detector).
  if (meta && meta->get<bool>("offline_source_drain_frame", false)) return;

  // Arrival-order frame number: the tapped IQ has no FFT counter, but the CHDR arrival index equals
  // signal_snipper's own IQ counter, so a per-channel arrival count keeps masks aligned to its ring.
  const uint64_t frame_number = ++frame_count_[ch];
  if (emit_stride_.get() > 1 && (frame_number % emit_stride_.get()) != 0) return;

  const int nfft = nfft_.get();          // model input width (fixed by the checkpoint)
  const int tile_rows = tile_rows_.get();
  if (nfft <= 0 || tile_rows <= 0) return;
  const bool downsample = real_time_downsample_.get();
  // Native mode FFTs at the model width; downsample mode FFTs wide (reproducing the app's dynamic
  // spectrogram) and later resizes the freq axis down to the model width inside the torch runtime.
  // downsample_fft_size <= 0 => auto-size from the actual receive rate (rate-adaptive: keeps ~512
  // rows/frame -> bounded 2-tile cost at any rate). Otherwise use the configured fixed size.
  // Receive rate from metadata: drives both the auto FFT sizing (downsample) and the training-power
  // level match (both paths). Fallback to the documented capture rate if the sidecar didn't stamp it.
  double rate_hz = 0.0;
  if (meta && meta->has_key("rx_sample_rate_hz")) rate_hz = meta->get<double>("rx_sample_rate_hz", 0.0);
  if (rate_hz <= 0.0 && meta && meta->has_key("sample_rate_hz")) rate_hz = meta->get<double>("sample_rate_hz", 0.0);
  if (rate_hz <= 0.0) rate_hz = 245.76e6;

  int fft_size = nfft;
  if (downsample) {
    fft_size = (downsample_fft_size_.get() > 0) ? std::max(nfft, downsample_fft_size_.get())
                                                : std::max(nfft, auto_fft_size(rate_hz));
  }

  // Total complex IQ samples in this frame -> whole fft_size-rows.
  const long total_samples = static_cast<long>(iq_tensor.Size(0)) * static_cast<long>(iq_tensor.Size(1));
  const int rows = static_cast<int>(total_samples / fft_size);
  if (rows <= 0) return;
  const int batch = (rows + tile_rows - 1) / tile_rows;

  auto& buf = channel_buffers_[ch];
  // Run all operator work on the IQ producer's stream so ordering with the upstream copy holds.
  const cudaStream_t stream = iq_stream;
  buf.ensure(static_cast<size_t>(rows), static_cast<size_t>(fft_size),
             static_cast<size_t>(tile_rows), static_cast<size_t>(batch));

  // ---- front-end: FFT (native nfft, or wide for downsample) -> dB -> [0,1] normalized ------
  auto* iq_ptr = reinterpret_cast<ft_dino_complex*>(iq_tensor.Data());
  auto iq_view = matx::make_tensor<ft_dino_complex>(iq_ptr, {rows, fft_size}, /*owning=*/false);
  auto spec_view = matx::make_tensor<ft_dino_complex>(
      reinterpret_cast<ft_dino_complex*>(buf.spec_device), {rows, fft_size}, /*owning=*/false);
  // Analysis window (freq axis) to suppress spectral leakage -- MUST match the training FFT window
  // (rfdata.frames_to_db). RMS-normalized so the noise-floor power is ~preserved. Filled once per
  // geometry; 'none' = legacy no-window.
  const std::string win = fft_window_.get();
  const bool use_window = (win != "none");
  if (use_window && buf.window_len != static_cast<size_t>(fft_size)) {
    std::vector<float> w(static_cast<size_t>(fft_size));
    double s2 = 0.0;
    for (int i = 0; i < fft_size; ++i) {
      const double x = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(fft_size);
      double v;
      if (win == "hamming")       v = 0.54 - 0.46 * std::cos(x);
      else if (win == "blackman") v = 0.42 - 0.5 * std::cos(x) + 0.08 * std::cos(2.0 * x);
      else                        v = 0.5 - 0.5 * std::cos(x);   // hann (default)
      w[i] = static_cast<float>(v); s2 += v * v;
    }
    const float rms = static_cast<float>(std::sqrt(s2 / std::max(1, fft_size)));
    for (int i = 0; i < fft_size; ++i) w[i] /= (rms > 1e-12f ? rms : 1.0f);
    throw_if_cuda_error(cudaMemcpyAsync(buf.window_device, w.data(), fft_size * sizeof(float),
                                        cudaMemcpyHostToDevice, stream), "window H2D");
    buf.window_len = static_cast<size_t>(fft_size);
  }
  // Per-row (batched) 1D FFT over the frequency axis; matx default norm matches torch (unnormalized).
  if (use_window) {
    auto w_view = matx::make_tensor<float>(buf.window_device, {fft_size}, /*owning=*/false);
    (spec_view = matx::fft(iq_view * w_view)).run(stream);
  } else {
    (spec_view = matx::fft(iq_view)).run(stream);
  }

  // Processing-gain correction: an unnormalized N-pt FFT scales a white-noise bin's power by N, so a
  // wide (downsample) FFT sits 10*log10(fft_size/nfft) dB hotter than the nfft-pt scale the model was
  // trained on. Shift vmin up by that offset (== subtract it from dB) to re-center the normalized image
  // onto the trained intensity distribution -- otherwise low-SNR signal near the noise floor is lost.
  // Zero for native mode (fft_size == nfft); derived from FFT sizes only, so it reproduces on any OTA
  // rate/geometry with no per-capture calibration.
  const double gain_offset_db = 10.0 * std::log10(static_cast<double>(fft_size) / static_cast<double>(nfft));
  // Training-power level match: the model uses a FIXED db_vmin/db_vmax, but an unnormalized nfft-pt
  // FFT's noise floor scales with the RBW (~ sample rate), so a deployment rate != the training rate
  // mis-levels the input under that fixed clip (a lower rate -> lower floor -> reads as background
  // shifted, and the passband can look like signal). Anchor the deployment floor to the training level
  // with a SINGLE scalar dB shift (no per-frequency term -> no shape change): 10*log10(rate/ref) + trim.
  // power_level_trim_db is ALWAYS applied (a clearly-signed manual knob: +dB raises vmin -> darkens
  // the floor -> fewer false positives; -dB brightens). The rate-derived RBW term is OPT-IN
  // (match_training_power_level) because it only captures the RBW/rate part and can be the wrong sign
  // when the deployment gain differs from the training captures (the usual dominant effect).
  double level_offset_db = power_level_trim_db_.get();
  if (match_training_power_level_.get()) {
    const double ref_rate = std::max(1.0, reference_sample_rate_hz_.get());
    level_offset_db += 10.0 * std::log10(rate_hz / ref_rate);
  }
  const double vmin_offset_db = gain_offset_db + level_offset_db;
  if (!level_log_emitted_) {
    std::fprintf(stderr, "[finetuned_dino_detector] power-level match %s: rate=%.4g MS/s ref=%.4g MS/s "
                 "-> level_offset=%.2f dB (gain_offset=%.2f dB, trim=%.2f dB); effective db_vmin=%.2f\n",
                 match_training_power_level_.get() ? "ON" : "OFF", rate_hz / 1e6,
                 reference_sample_rate_hz_.get() / 1e6, level_offset_db, gain_offset_db,
                 power_level_trim_db_.get(), db_vmin_.get() + vmin_offset_db);
    level_log_emitted_ = true;
  }
  const float inv_span = 1.0f / std::max(static_cast<float>(db_vmax_.get() - db_vmin_.get()), 1e-6f);
  const long spec_total = static_cast<long>(rows) * fft_size;
  const int threads = 256;
  const int spec_blocks = static_cast<int>((spec_total + threads - 1) / threads);
  const auto* spec_f2 = reinterpret_cast<const float2*>(buf.spec_device);

  if (flatten_noise_floor_.get()) {
    // Split the fused power->dB->clip so we can flatten the per-frequency floor on the dB image first.
    // gain_offset is subtracted here (trained-scale dB), so the clip below uses the raw db_vmin.
    ft_power_to_db_shift_kernel<<<spec_blocks, threads, 0, stream>>>(
        spec_f2, buf.db_device, rows, fft_size, static_cast<float>(gain_offset_db));
    // Bandwidth-invariant smoothing: sigma tracks a fixed fraction of the band, so the same config
    // works at any sample rate (the filter rolloff occupies a fixed fraction of Nyquist).
    const float sigma = std::max(2.0f, static_cast<float>(flatten_smooth_frac_.get()) * fft_size);
    const int smooth_radius = std::max(1, static_cast<int>(std::ceil(sigma * 1.5f)));
    const int col_blocks = (fft_size + threads - 1) / threads;
    const float q = static_cast<float>(flatten_reference_q_.get() / 100.0);
    // Pass 1: uncapped per-freq floor -> smooth -> reference. Pass 2 (if signal-cap on): recompute the
    // floor with signal influence capped at reference+cap so strong bins don't inflate their own floor.
    ft_col_mean_kernel<<<fft_size, threads, 0, stream>>>(buf.db_device, rows, fft_size, nullptr, 0.0f,
                                                         buf.col_stat_device);
    ft_smooth_cols_kernel<<<col_blocks, threads, 0, stream>>>(buf.col_stat_device, fft_size, smooth_radius,
                                                              sigma, buf.col_smooth_device);
    ft_frontend_reference_kernel<<<1, threads, 0, stream>>>(buf.col_smooth_device, fft_size, q,
                                                            buf.frontend_reference_device);
    if (flatten_signal_cap_db_.get() > 0.0) {
      ft_col_mean_kernel<<<fft_size, threads, 0, stream>>>(
          buf.db_device, rows, fft_size, buf.frontend_reference_device,
          static_cast<float>(flatten_signal_cap_db_.get()), buf.col_stat_device);
      ft_smooth_cols_kernel<<<col_blocks, threads, 0, stream>>>(buf.col_stat_device, fft_size, smooth_radius,
                                                                sigma, buf.col_smooth_device);
      ft_frontend_reference_kernel<<<1, threads, 0, stream>>>(buf.col_smooth_device, fft_size, q,
                                                              buf.frontend_reference_device);
    }
    ft_frontend_correction_kernel<<<spec_blocks, threads, 0, stream>>>(
        buf.db_device, rows, fft_size, buf.col_smooth_device, buf.frontend_reference_device,
        static_cast<float>(flatten_max_boost_db_.get()));
    ft_clip_normalize_kernel<<<spec_blocks, threads, 0, stream>>>(
        buf.db_device, buf.normalized_device, spec_total,
        static_cast<float>(db_vmin_.get() + level_offset_db), inv_span);  // gain already subtracted in power_to_db
    throw_if_cuda_error(cudaGetLastError(), "flatten frontend kernels");
    if (!flatten_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] per-freq floor flatten ON (q=%.0f frac=%.4f "
                   "-> sigma=%.1f bins @ fft=%d, max_boost=%.1f dB signal_cap=%.1f dB)\n",
                   flatten_reference_q_.get(), flatten_smooth_frac_.get(), sigma, fft_size,
                   flatten_max_boost_db_.get(), flatten_signal_cap_db_.get());
      flatten_log_emitted_ = true;
    }
  } else {
    // Original fused path (validated): fold the gain + power-level offsets into vmin, single kernel.
    const float vmin = static_cast<float>(db_vmin_.get() + vmin_offset_db);
    power_db_normalize_kernel<<<spec_blocks, threads, 0, stream>>>(
        spec_f2, buf.normalized_device, rows, fft_size, vmin, inv_span);
    throw_if_cuda_error(cudaGetLastError(), "power_db_normalize kernel");
  }

  if (!(runtime_ && runtime_->loaded())) {
    if (!startup_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] inference unavailable (model not loaded)\n");
      startup_log_emitted_ = true;
    }
    return;  // don't emit a bogus mask
  }

  // Filled by whichever path runs; emitted once at the end. width = mask columns on the native grid.
  uint8_t* emit_mask = nullptr;
  int mask_width = 0;

  if (downsample) {
    // ---- downsample path: resize freq -> model width, tile, infer, threshold, upsample (torch) ----
    const size_t mask_bytes = static_cast<size_t>(rows) * fft_size * sizeof(uint8_t);
    throw_if_cuda_error(cudaMalloc(&emit_mask, mask_bytes), "malloc emit mask");
    double inference_ms = 0.0;
    const bool ok = runtime_->forward_downsampled(buf.normalized_device, rows, fft_size, tile_rows,
                                                  nfft, static_cast<float>(threshold_.get()),
                                                  emit_mask, stream, &inference_ms);
    if (!ok) { cudaFree(emit_mask); return; }
    ++inference_samples_;
    inference_ms_ewma_ += (inference_ms - inference_ms_ewma_) / static_cast<double>(inference_samples_);
    if (inference_samples_ == 1 || (inference_samples_ % 32) == 0) {
      std::fprintf(stderr, "[finetuned_dino_detector] downsample: FFT %d -> model %d (gain corr "
                   "%.2f dB), %d rows -> %d tiles; inference %.2f ms (mean %.2f ms over %llu frames)\n",
                   fft_size, nfft, gain_offset_db, rows, batch, inference_ms, inference_ms_ewma_,
                   static_cast<unsigned long long>(inference_samples_));
    }
    mask_width = fft_size;   // emitted on the wide (native display) grid
  } else {

  // ---- tile into [B,1,tile_rows,nfft] (pad the last tile with zeros) ------------------------
  const size_t row_bytes = static_cast<size_t>(nfft) * sizeof(float);
  const size_t tile_stride = static_cast<size_t>(tile_rows) * nfft;
  throw_if_cuda_error(cudaMemsetAsync(buf.tile_batch_device, 0,
                                      static_cast<size_t>(batch) * tile_stride * sizeof(float), stream),
                      "memset tiles");
  for (int b = 0; b < batch; ++b) {
    const int r0 = b * tile_rows;
    const int valid = std::min(tile_rows, rows - r0);
    if (valid <= 0) break;
    throw_if_cuda_error(
        cudaMemcpyAsync(buf.tile_batch_device + static_cast<size_t>(b) * tile_stride,
                        buf.normalized_device + static_cast<size_t>(r0) * nfft,
                        static_cast<size_t>(valid) * row_bytes, cudaMemcpyDeviceToDevice, stream),
        "tile copy");
  }

  // ---- inference ----------------------------------------------------------------------------
  const bool ok = runtime_->forward(buf.tile_batch_device, batch, tile_rows, nfft,
                                    buf.logits_device, stream);
  if (!ok) return;  // don't emit a bogus mask

  // ---- threshold + stitch tiles back to a native (rows x nfft) mask -------------------------
  const long tile_total = static_cast<long>(batch) * tile_stride;
  sigmoid_threshold_kernel<<<static_cast<int>((tile_total + threads - 1) / threads), threads, 0, stream>>>(
      buf.logits_device, buf.tile_mask_device, tile_total, static_cast<float>(threshold_.get()));
  throw_if_cuda_error(cudaGetLastError(), "sigmoid_threshold kernel");

  // Per-emit device mask buffer, ownership handed to the message (downstream copies it out).
  const size_t mask_bytes = static_cast<size_t>(rows) * nfft * sizeof(uint8_t);
  throw_if_cuda_error(cudaMalloc(&emit_mask, mask_bytes), "malloc emit mask");
  const size_t mask_row_bytes = static_cast<size_t>(nfft) * sizeof(uint8_t);
  for (int b = 0; b < batch; ++b) {
    const int r0 = b * tile_rows;
    const int valid = std::min(tile_rows, rows - r0);
    if (valid <= 0) break;
    throw_if_cuda_error(
        cudaMemcpyAsync(emit_mask + static_cast<size_t>(r0) * nfft,
                        buf.tile_mask_device + static_cast<size_t>(b) * tile_stride,
                        static_cast<size_t>(valid) * mask_row_bytes, cudaMemcpyDeviceToDevice, stream),
        "stitch copy");
  }
  // Make the mask valid for the downstream's (blocking) copy out.
  throw_if_cuda_error(cudaStreamSynchronize(stream), "stream sync before emit");
    mask_width = nfft;
  }  // end native path

  holoscan::ops::DetectorMaskMessage mask_msg;
  mask_msg.width = mask_width;
  mask_msg.height = rows;
  mask_msg.channel = channel_number;
  mask_msg.frame_number = frame_number;
  mask_msg.device_pixels =
      std::shared_ptr<uint8_t>(emit_mask, [](uint8_t* p) { cudaFree(p); });
  if (meta) {
    mask_msg.file_offset_complex    = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
    mask_msg.data_end_complex       = meta->get<uint64_t>("offline_source_data_end_complex", 0);
    mask_msg.frame_end_complex      = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
    mask_msg.complex_samples_read   = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
    mask_msg.complex_samples_padded = meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
  }
  op_output.emit(mask_msg, "mask_out");
  if (meta) meta->set("finetuned_dino_mask_emitted", true);
  ++compute_count_;
}

}  // namespace holoscan::ops

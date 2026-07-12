// SPDX-FileCopyrightText: 2026 The University of Texas at Austin
//
// SPDX-License-Identifier: Apache-2.0
#include "power_detection_tuned.hpp"

// Shared detector output contract (see cuda_dino_detector / power_detection).
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace holoscan::ops {

namespace {

constexpr float kPowerEps = 1e-20f;  // linear-power floor (avoid log(0))

void throw_if_cuda_error(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("power_detection_tuned CUDA error during ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

void throw_if_cufft_error(cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("power_detection_tuned cuFFT error during ") + what +
                             ": code=" + std::to_string(static_cast<int>(status)));
  }
}

std::shared_ptr<uint8_t> make_owned_device_u8(size_t count) {
  uint8_t* raw = nullptr;
  throw_if_cuda_error(cudaMalloc(&raw, count * sizeof(uint8_t)), "cudaMalloc(mask)");
  return std::shared_ptr<uint8_t>(raw, [](uint8_t* p) {
    if (p != nullptr) {
      cudaFree(p);
    }
  });
}

// Window kinds. Keep in sync with the string parsed in compute().
enum class WindowKind { kNone = 0, kHann = 1, kBlackmanHarris = 2 };

// Fill the analysis window taps (one thread per tap).
__global__ void fill_window_kernel(float* __restrict__ w, int n, int kind) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (n <= 1) {
    w[i] = 1.0f;
    return;
  }
  const float t = static_cast<float>(i) / static_cast<float>(n - 1);
  const float two_pi = 6.283185307179586f;
  if (kind == static_cast<int>(WindowKind::kHann)) {
    w[i] = 0.5f - 0.5f * cosf(two_pi * t);
  } else if (kind == static_cast<int>(WindowKind::kBlackmanHarris)) {
    const float a0 = 0.35875f, a1 = 0.48829f, a2 = 0.14128f, a3 = 0.01168f;
    w[i] = a0 - a1 * cosf(two_pi * t) + a2 * cosf(2.0f * two_pi * t) -
           a3 * cosf(3.0f * two_pi * t);
  } else {
    w[i] = 1.0f;
  }
}

// Multiply each sample by the per-frequency-sample window tap.
__global__ void apply_window_kernel(const complex* __restrict__ in,
                                    const float* __restrict__ w,
                                    complex* __restrict__ out,
                                    int rows,
                                    int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const size_t idx = static_cast<size_t>(row) * cols + col;
  const float g = w[col];
  const complex v = in[idx];
  out[idx] = complex(v.real() * g, v.imag() * g);
}

// Linear power |X|^2 with an fftshift along frequency (columns), and its dB.
__global__ void power_lin_db_shift_kernel(const complex* __restrict__ in,
                                          float* __restrict__ out_lin,
                                          float* __restrict__ out_db,
                                          int rows,
                                          int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const int src_col = (col + cols / 2) % cols;
  const complex v = in[static_cast<size_t>(row) * cols + src_col];
  const float re = v.real();
  const float im = v.imag();
  const float power = re * re + im * im + kPowerEps;
  const size_t dst = static_cast<size_t>(row) * cols + col;
  out_lin[dst] = power;
  out_db[dst] = 10.0f * log10f(power);
}

// 2-D CFAR on linear power. For each cell-under-test (CUT) a reference region is
// formed from a (train x train) box in frequency and time, excluding a guard box
// around the CUT. The threshold multiplier alpha is derived from the target Pfa
// for exponentially distributed noise power: alpha = N*(Pfa^(-1/N) - 1), giving a
// constant false-alarm rate. CA uses the whole reference mean; GO/SO take the
// greatest/smallest of the leading (higher-freq) and lagging (lower-freq) halves.
//   variant: 0 = CA, 1 = GO, 2 = SO.
__global__ void cfar_kernel(const float* __restrict__ lin,
                            uint8_t* __restrict__ mask,
                            int rows,
                            int cols,
                            int guard_f,
                            int train_f,
                            int guard_t,
                            int train_t,
                            float pfa,
                            int variant,
                            int dc_center,
                            int dc_notch) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const size_t idx = static_cast<size_t>(row) * cols + col;

  if (dc_notch > 0 && abs(col - dc_center) <= dc_notch) {
    mask[idx] = 0;
    return;
  }

  double sum_all = 0.0;
  int cnt_all = 0;
  double sum_lead = 0.0;  // dc > 0 (higher frequency)
  int cnt_lead = 0;
  double sum_lag = 0.0;   // dc < 0 (lower frequency)
  int cnt_lag = 0;

  for (int dr = -(guard_t + train_t); dr <= (guard_t + train_t); ++dr) {
    const int r = row + dr;
    if (r < 0 || r >= rows) {
      continue;
    }
    for (int dc = -(guard_f + train_f); dc <= (guard_f + train_f); ++dc) {
      // Skip the guard box (including the CUT itself).
      if (abs(dr) <= guard_t && abs(dc) <= guard_f) {
        continue;
      }
      const int c = col + dc;
      if (c < 0 || c >= cols) {
        continue;
      }
      const double x = lin[static_cast<size_t>(r) * cols + c];
      sum_all += x;
      ++cnt_all;
      if (dc > 0) {
        sum_lead += x;
        ++cnt_lead;
      } else if (dc < 0) {
        sum_lag += x;
        ++cnt_lag;
      }
    }
  }

  uint8_t flagged = 0;
  if (cnt_all >= 4) {
    // Noise-power estimate per the requested variant.
    double noise_est = sum_all / cnt_all;
    if (variant == 1 || variant == 2) {
      const bool have_lead = cnt_lead > 0;
      const bool have_lag = cnt_lag > 0;
      const double mean_lead = have_lead ? sum_lead / cnt_lead : noise_est;
      const double mean_lag = have_lag ? sum_lag / cnt_lag : noise_est;
      if (variant == 1) {  // GO: greatest-of -> conservative on sloping floors
        noise_est = fmax(mean_lead, mean_lag);
      } else {             // SO: smallest-of -> bites into extended-target edges
        noise_est = fmin(mean_lead, mean_lag);
      }
    }
    // alpha = N*(Pfa^(-1/N) - 1); N = number of reference cells actually used.
    const double n = static_cast<double>(cnt_all);
    const double alpha = n * (pow(static_cast<double>(pfa), -1.0 / n) - 1.0);
    const double threshold = alpha * noise_est;
    flagged = (static_cast<double>(lin[idx]) > threshold) ? 255 : 0;
  }
  mask[idx] = flagged;
}

// Update the adaptive per-bin temporal noise floor (dB). One thread per bin sums
// this frame's dB over all time rows, then blends into the EMA -- but only if the
// bin's per-frame mean is not itself signal-like (sigma-clip), so persistent /
// wideband signals do not corrupt the floor. On the first update the EMA is
// seeded directly from the frame statistics.
__global__ void temporal_update_kernel(const float* __restrict__ db,
                                        float* __restrict__ ema_mean,
                                        float* __restrict__ ema_var,
                                        int rows,
                                        int cols,
                                        float alpha,
                                        float clip_z,
                                        float min_std,
                                        int initialized) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  double s = 0.0;
  double ss = 0.0;
  for (int r = 0; r < rows; ++r) {
    const double x = db[static_cast<size_t>(r) * cols + col];
    s += x;
    ss += x * x;
  }
  const double dn = static_cast<double>(rows);
  const float frame_mean = static_cast<float>(s / dn);
  const double var = ss / dn - (s / dn) * (s / dn);
  const float frame_var = static_cast<float>(var > 0.0 ? var : 0.0);

  if (!initialized) {
    ema_mean[col] = frame_mean;
    ema_var[col] = frame_var;
    return;
  }

  // Sigma-clip: freeze the floor for this bin if the frame looks signal-occupied.
  const float cur_std = sqrtf(fmaxf(ema_var[col], min_std * min_std));
  if (frame_mean > ema_mean[col] + clip_z * cur_std) {
    return;
  }
  ema_mean[col] = (1.0f - alpha) * ema_mean[col] + alpha * frame_mean;
  ema_var[col] = (1.0f - alpha) * ema_var[col] + alpha * frame_var;
}

// Detect exceedance vs. the adaptive per-bin temporal floor (z-score in dB).
// OR-composites into an existing mask when `combine` is set.
__global__ void temporal_detect_kernel(const float* __restrict__ db,
                                        const float* __restrict__ ema_mean,
                                        const float* __restrict__ ema_var,
                                        uint8_t* __restrict__ mask,
                                        int rows,
                                        int cols,
                                        float zthr,
                                        float min_std,
                                        int combine,
                                        int dc_center,
                                        int dc_notch) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const size_t idx = static_cast<size_t>(row) * cols + col;
  if (dc_notch > 0 && abs(col - dc_center) <= dc_notch) {
    if (!combine) {
      mask[idx] = 0;
    }
    return;
  }
  const float std = sqrtf(fmaxf(ema_var[col], min_std * min_std));
  const float z = (db[idx] - ema_mean[col]) / std;
  const uint8_t hit = (z > zthr) ? 255 : 0;
  if (combine) {
    if (hit) {
      mask[idx] = 255;
    }
  } else {
    mask[idx] = hit;
  }
}

int parse_window_kind(const std::string& s) {
  if (s == "none") {
    return static_cast<int>(WindowKind::kNone);
  }
  if (s == "hann") {
    return static_cast<int>(WindowKind::kHann);
  }
  return static_cast<int>(WindowKind::kBlackmanHarris);  // default
}

int parse_cfar_variant(const std::string& s) {
  if (s == "ca") {
    return 0;
  }
  if (s == "so") {
    return 2;
  }
  return 1;  // "go" default
}

}  // namespace

void PowerDetectionTuned::setup(OperatorSpec& spec) {
  spec.input<power_detection_tuned_in_t>("in", holoscan::IOSpec::IOSize{16});
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(burst_size_, "burst_size", "Burst size",
             "Complex samples per burst (FFT length). Must match the raw-IQ producer.");
  spec.param(num_bursts_, "num_bursts", "Number of bursts",
             "Time rows processed per frame. Must match the raw-IQ producer.");
  spec.param(num_channels_, "num_channels", "Number of channels",
             "Pipeline channel count (one-to-one routing validation).", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter",
             "Channel index this operator instance handles.", 0);

  spec.param(mode_, "mode", "Detection mode",
             "'cfar' (2-D CFAR only), 'temporal' (adaptive per-bin floor only), or "
             "'combined' (logical OR of both -- strongest classical config).",
             std::string("combined"));
  spec.param(window_type_, "window_type", "FFT window",
             "Analysis window applied before the FFT: 'blackman_harris', 'hann', or 'none'.",
             std::string("blackman_harris"));
  spec.param(cfar_variant_, "cfar_variant", "CFAR variant",
             "'ca' (cell-averaging), 'go' (greatest-of, robust on sloping floors), or "
             "'so' (smallest-of, bites into extended-target edges).",
             std::string("go"));
  spec.param(pfa_, "pfa", "CFAR false-alarm probability",
             "Target per-cell false-alarm probability; sets the CFAR multiplier "
             "alpha = N*(Pfa^(-1/N) - 1). System-agnostic (no dB threshold).",
             1e-3f);
  spec.param(guard_freq_, "guard_freq", "Guard bins (frequency)",
             "Guard bins each side of the CUT in frequency (covers window main-lobe width).", 4);
  spec.param(train_freq_, "train_freq", "Training bins (frequency)",
             "Training bins each side of the CUT in frequency.", 24);
  spec.param(guard_time_, "guard_time", "Guard rows (time)",
             "Guard rows each side of the CUT in time.", 1);
  spec.param(train_time_, "train_time", "Training rows (time)",
             "Training rows each side of the CUT in time.", 4);
  spec.param(temporal_zscore_, "temporal_zscore", "Temporal z-score",
             "Detection threshold (sigma) against the adaptive per-bin temporal floor.", 5.0f);
  spec.param(temporal_alpha_, "temporal_alpha", "Temporal EMA rate",
             "Exponential-moving-average update rate for the temporal noise floor.", 0.05f);
  spec.param(temporal_clip_z_, "temporal_clip_z", "Temporal sigma-clip",
             "Freeze a bin's temporal floor when this frame's per-bin mean exceeds "
             "floor + clip_z*sigma (signal-occupied bin).", 3.0f);
  spec.param(min_std_db_, "min_std_db", "Minimum std (dB)",
             "Floor on the temporal std estimate to avoid divide-by-noise.", 0.5f);
  spec.param(warmup_frames_, "warmup_frames", "Warm-up frames",
             "Frames spent learning the temporal floor before it is allowed to detect.", 2);
  spec.param(dc_notch_bins_, "dc_notch_bins", "DC notch bins",
             "Bins each side of band center forced to no-detect (DC / LO leakage).", 4);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one mask every N frames.", 1);
}

void PowerDetectionTuned::initialize() {
  holoscan::Operator::initialize();
  if (burst_size_.get() <= 0 || num_bursts_.get() <= 0) {
    throw std::runtime_error("power_detection_tuned requires positive burst_size and num_bursts");
  }
}

void PowerDetectionTuned::ensure_allocated(cudaStream_t stream) {
  if (state_.allocated) {
    return;
  }
  const size_t rows = static_cast<size_t>(num_bursts_.get());
  const size_t cols = static_cast<size_t>(burst_size_.get());
  const size_t cells = rows * cols;

  throw_if_cuda_error(cudaMalloc(&state_.window, cols * sizeof(float)), "cudaMalloc(window)");
  throw_if_cuda_error(cudaMalloc(&state_.windowed, cells * sizeof(complex)), "cudaMalloc(windowed)");
  throw_if_cuda_error(cudaMalloc(&state_.fft_out, cells * sizeof(complex)), "cudaMalloc(fft_out)");
  throw_if_cuda_error(cudaMalloc(&state_.power_lin, cells * sizeof(float)), "cudaMalloc(power_lin)");
  throw_if_cuda_error(cudaMalloc(&state_.power_db, cells * sizeof(float)), "cudaMalloc(power_db)");
  throw_if_cuda_error(cudaMalloc(&state_.ema_mean, cols * sizeof(float)), "cudaMalloc(ema_mean)");
  throw_if_cuda_error(cudaMalloc(&state_.ema_var, cols * sizeof(float)), "cudaMalloc(ema_var)");

  // Precompute the analysis window once.
  const int wk = parse_window_kind(window_type_.get());
  const int block1d = 256;
  const int wgrid = (static_cast<int>(cols) + block1d - 1) / block1d;
  fill_window_kernel<<<wgrid, block1d, 0, stream>>>(state_.window, static_cast<int>(cols), wk);
  throw_if_cuda_error(cudaGetLastError(), "fill_window_kernel launch");

  int fft_dims[1] = {burst_size_.get()};
  throw_if_cufft_error(cufftPlanMany(&state_.fft_plan, 1, fft_dims, fft_dims, 1, burst_size_.get(),
                                     fft_dims, 1, burst_size_.get(), CUFFT_C2C, num_bursts_.get()),
                       "cufftPlanMany");
  state_.fft_plan_initialized = true;
  state_.allocated = true;
}

void PowerDetectionTuned::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  auto maybe_input = op_input.receive<power_detection_tuned_in_t>("in");
  if (!maybe_input) {
    return;
  }
  auto input = maybe_input.value();
  auto& in_tensor = std::get<0>(input);
  cudaStream_t stream = std::get<1>(input);

  const int rows = num_bursts_.get();
  const int cols = burst_size_.get();
  if (in_tensor.Size(0) < rows || in_tensor.Size(1) < cols) {
    HOLOSCAN_LOG_WARN(
        "power_detection_tuned: input {}x{} smaller than configured {}x{}; skipping frame.",
        in_tensor.Size(0), in_tensor.Size(1), rows, cols);
    return;
  }

  ensure_allocated(stream);

  auto meta = metadata();
  const uint16_t channel_number =
      meta ? meta->get<uint16_t>("channel_number", static_cast<uint16_t>(channel_filter_.get()))
           : static_cast<uint16_t>(channel_filter_.get());

  const dim3 block2d(32, 8);
  const dim3 grid2d((cols + block2d.x - 1) / block2d.x, (rows + block2d.y - 1) / block2d.y);
  const int block1d = 256;
  const int grid1d_cols = (cols + block1d - 1) / block1d;

  // 1) Window + FFT of the raw IQ (this baseline owns its own transform).
  apply_window_kernel<<<grid2d, block2d, 0, stream>>>(in_tensor.Data(), state_.window,
                                                      state_.windowed, rows, cols);
  throw_if_cuda_error(cudaGetLastError(), "apply_window_kernel launch");
  throw_if_cufft_error(cufftSetStream(state_.fft_plan, stream), "cufftSetStream");
  throw_if_cufft_error(cufftExecC2C(state_.fft_plan,
                                    reinterpret_cast<cufftComplex*>(state_.windowed),
                                    reinterpret_cast<cufftComplex*>(state_.fft_out),
                                    CUFFT_FORWARD),
                       "cufftExecC2C");

  // 2) Linear power + dB spectrogram (fftshifted along frequency).
  power_lin_db_shift_kernel<<<grid2d, block2d, 0, stream>>>(state_.fft_out, state_.power_lin,
                                                            state_.power_db, rows, cols);
  throw_if_cuda_error(cudaGetLastError(), "power_lin_db_shift_kernel launch");

  ++frame_number_;
  const std::string mode = mode_.get();
  const bool use_cfar = (mode == "cfar" || mode == "combined");
  const bool use_temporal = (mode == "temporal" || mode == "combined");

  // Update the adaptive temporal floor every frame (independent of emit stride).
  if (use_temporal) {
    temporal_update_kernel<<<grid1d_cols, block1d, 0, stream>>>(
        state_.power_db, state_.ema_mean, state_.ema_var, rows, cols, temporal_alpha_.get(),
        temporal_clip_z_.get(), min_std_db_.get(), state_.ema_initialized ? 1 : 0);
    throw_if_cuda_error(cudaGetLastError(), "temporal_update_kernel launch");
    state_.ema_initialized = true;
  }

  const int stride = std::max(1, emit_stride_.get());
  const bool warming_up =
      use_temporal && !use_cfar && frame_number_ <= static_cast<uint64_t>(warmup_frames_.get());
  const bool emit_this_frame = (frame_number_ % static_cast<uint64_t>(stride)) == 0 && !warming_up;
  if (!emit_this_frame) {
    if (meta) {
      meta->set("power_detection_tuned_emitted", false);
      meta->set("power_detection_tuned_warming_up", warming_up);
    }
    return;
  }

  // 3) Detection -> owned mask buffer.
  const size_t cells = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  auto mask_device = make_owned_device_u8(cells);
  const int dc_center = cols / 2;
  const int dc_notch = std::max(0, dc_notch_bins_.get());

  if (use_cfar) {
    cfar_kernel<<<grid2d, block2d, 0, stream>>>(
        state_.power_lin, mask_device.get(), rows, cols, std::max(0, guard_freq_.get()),
        std::max(1, train_freq_.get()), std::max(0, guard_time_.get()),
        std::max(0, train_time_.get()), pfa_.get(), parse_cfar_variant(cfar_variant_.get()),
        dc_center, dc_notch);
    throw_if_cuda_error(cudaGetLastError(), "cfar_kernel launch");
  } else {
    throw_if_cuda_error(cudaMemsetAsync(mask_device.get(), 0, cells, stream), "cudaMemset(mask)");
  }

  // Temporal detection: OR into the CFAR mask (combined) or write it (temporal-only).
  const bool temporal_active =
      use_temporal && frame_number_ > static_cast<uint64_t>(warmup_frames_.get());
  if (temporal_active) {
    temporal_detect_kernel<<<grid2d, block2d, 0, stream>>>(
        state_.power_db, state_.ema_mean, state_.ema_var, mask_device.get(), rows, cols,
        temporal_zscore_.get(), min_std_db_.get(), use_cfar ? 1 : 0, dc_center, dc_notch);
    throw_if_cuda_error(cudaGetLastError(), "temporal_detect_kernel launch");
  }

  throw_if_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize before emit");

  DetectorMaskMessage mask_msg;
  mask_msg.device_pixels = std::move(mask_device);
  mask_msg.width = cols;   // frequency axis
  mask_msg.height = rows;  // time axis
  mask_msg.channel = static_cast<int>(channel_number);
  mask_msg.frame_number = frame_number_;
  if (meta) {
    mask_msg.file_offset_complex = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
    mask_msg.data_end_complex = meta->get<uint64_t>("offline_source_data_end_complex", 0);
    mask_msg.frame_end_complex = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
    mask_msg.complex_samples_read = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
    mask_msg.complex_samples_padded = meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
  }

  op_output.emit(mask_msg, "mask_out");
  ++detections_emitted_;

  if (meta) {
    meta->set("power_detection_tuned_emitted", true);
    meta->set("power_detection_tuned_mode", mode);
    meta->set("power_detection_tuned_pfa", pfa_.get());
    meta->set("power_detection_tuned_frame_number", frame_number_);
  }
}

void PowerDetectionTuned::free_device_state() {
  if (state_.fft_plan_initialized) {
    cufftDestroy(state_.fft_plan);
    state_.fft_plan = 0;
    state_.fft_plan_initialized = false;
  }
  auto free_ptr = [](auto*& p) {
    if (p != nullptr) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_ptr(state_.window);
  free_ptr(state_.windowed);
  free_ptr(state_.fft_out);
  free_ptr(state_.power_lin);
  free_ptr(state_.power_db);
  free_ptr(state_.ema_mean);
  free_ptr(state_.ema_var);
  state_.allocated = false;
  state_.ema_initialized = false;
}

void PowerDetectionTuned::stop() {
  HOLOSCAN_LOG_INFO(
      "power_detection_tuned ch={} processed_frames={} emitted_masks={} mode={} variant={}",
      channel_filter_.get(), frame_number_, detections_emitted_, mode_.get(), cfar_variant_.get());
  free_device_state();
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

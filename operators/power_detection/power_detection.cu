// SPDX-FileCopyrightText: 2026 The University of Texas at Austin
//
// SPDX-License-Identifier: Apache-2.0
#include "power_detection.hpp"

// DetectorMaskMessage is the shared detector output contract used by the DINO
// detectors and the downstream visualization/comparison tooling. Pull it in via
// the same relative path the cuda_dino_detector uses so this baseline is a
// drop-in peer.
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace holoscan::ops {

namespace {

constexpr float kPowerEps = 1e-12f;

void throw_if_cuda_error(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("power_detection CUDA error during ") + what + ": " +
                             cudaGetErrorString(status));
  }
}

void throw_if_cufft_error(cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("power_detection cuFFT error during ") + what +
                             ": code=" + std::to_string(static_cast<int>(status)));
  }
}

// Allocate a fresh device buffer whose ownership is handed downstream through
// the DetectorMaskMessage. Downstream stores masks in a pending-map, so the
// buffer must be owned (not a reused scratch buffer).
std::shared_ptr<uint8_t> make_owned_device_u8(size_t count) {
  uint8_t* raw = nullptr;
  throw_if_cuda_error(cudaMalloc(&raw, count * sizeof(uint8_t)), "cudaMalloc(mask)");
  return std::shared_ptr<uint8_t>(raw, [](uint8_t* p) {
    if (p != nullptr) {
      cudaFree(p);
    }
  });
}

// Power spectrum in dB with an fftshift applied along frequency (columns).
// out[r, c] = 10*log10(|in[r, (c + cols/2) % cols]|^2 + eps)
__global__ void power_db_shift_kernel(const complex* __restrict__ in,
                                      float* __restrict__ out_db,
                                      int rows,
                                      int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const int src_col = (col + cols / 2) % cols;
  const complex v = in[row * cols + src_col];
  const float re = v.real();
  const float im = v.imag();
  const float power = re * re + im * im + kPowerEps;
  out_db[row * cols + col] = 10.0f * log10f(power);
}

// Cell-averaging CFAR-style z-score threshold along frequency. For every cell,
// estimate the local mean/std from training bins on both sides of the cell
// under test, skipping the guard bins immediately adjacent to it. Flags cells
// whose power exceeds the local mean by more than `zthr` standard deviations.
// Stateless (uses only the current frame) and free of absolute dB thresholds.
__global__ void cfar_zscore_kernel(const float* __restrict__ db,
                                    uint8_t* __restrict__ mask,
                                    int rows,
                                    int cols,
                                    int win,
                                    int guard,
                                    float zthr,
                                    float min_std) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }

  const float* row_ptr = db + static_cast<size_t>(row) * cols;
  double sum = 0.0;
  double sumsq = 0.0;
  int count = 0;

  // Left training window.
  const int left_hi = col - guard - 1;
  const int left_lo = col - guard - win;
  for (int c = left_lo; c <= left_hi; ++c) {
    if (c >= 0 && c < cols) {
      const float x = row_ptr[c];
      sum += x;
      sumsq += static_cast<double>(x) * x;
      ++count;
    }
  }
  // Right training window.
  const int right_lo = col + guard + 1;
  const int right_hi = col + guard + win;
  for (int c = right_lo; c <= right_hi; ++c) {
    if (c >= 0 && c < cols) {
      const float x = row_ptr[c];
      sum += x;
      sumsq += static_cast<double>(x) * x;
      ++count;
    }
  }

  uint8_t flagged = 0;
  if (count >= 2) {
    const double mean = sum / count;
    const double var = sumsq / count - mean * mean;
    const double std = sqrt(var > 0.0 ? var : 0.0);
    const double eff_std = std > min_std ? std : static_cast<double>(min_std);
    const double z = (static_cast<double>(row_ptr[col]) - mean) / eff_std;
    flagged = (z > zthr) ? 255 : 0;
  }
  mask[static_cast<size_t>(row) * cols + col] = flagged;
}

// Fold one frame's per-bin dB values into the running baseline accumulators.
// One thread per frequency bin sums over all time rows (no atomics needed).
__global__ void baseline_accumulate_kernel(const float* __restrict__ db,
                                           double* __restrict__ baseline_sum,
                                           double* __restrict__ baseline_sumsq,
                                           int rows,
                                           int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  double s = 0.0;
  double ss = 0.0;
  for (int r = 0; r < rows; ++r) {
    const float x = db[static_cast<size_t>(r) * cols + col];
    s += x;
    ss += static_cast<double>(x) * x;
  }
  baseline_sum[col] += s;
  baseline_sumsq[col] += ss;
}

// Detect exceedance against the frozen per-bin baseline in z-score units.
__global__ void baseline_detect_kernel(const float* __restrict__ db,
                                       const double* __restrict__ baseline_sum,
                                       const double* __restrict__ baseline_sumsq,
                                       double nsamp,
                                       uint8_t* __restrict__ mask,
                                       int rows,
                                       int cols,
                                       float zthr,
                                       float min_std) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const double mean = baseline_sum[col] / nsamp;
  const double var = baseline_sumsq[col] / nsamp - mean * mean;
  const double std = sqrt(var > 0.0 ? var : 0.0);
  const double eff_std = std > min_std ? std : static_cast<double>(min_std);
  const double x = db[static_cast<size_t>(row) * cols + col];
  const double z = (x - mean) / eff_std;
  mask[static_cast<size_t>(row) * cols + col] = (z > zthr) ? 255 : 0;
}

}  // namespace

void PowerDetection::setup(OperatorSpec& spec) {
  spec.input<power_detection_in_t>("in", holoscan::IOSpec::IOSize{16});
  // Match the DINO detectors' optional mask output so downstream consumers are shared.
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(burst_size_, "burst_size", "Burst size",
             "Number of complex samples per burst (FFT length). Must match the raw-IQ producer.");
  spec.param(num_bursts_, "num_bursts", "Number of bursts",
             "Number of bursts (time rows) processed per frame. Must match the raw-IQ producer.");
  spec.param(num_channels_, "num_channels", "Number of channels",
             "Number of channels in the pipeline (used for one-to-one routing validation).", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter",
             "Channel index this operator instance is responsible for.", 0);

  spec.param(threshold_mode_, "threshold_mode", "Threshold mode",
             "Detection rule: 'moving_average' (per-frame CA-CFAR z-score across frequency) or "
             "'baseline' (per-bin exceedance vs. a noise floor learned from the first N frames).",
             std::string("moving_average"));
  spec.param(zscore_threshold_, "zscore_threshold", "Z-score threshold",
             "Flag a cell when its power exceeds the local/baseline mean by this many standard "
             "deviations (N-sigma). System-agnostic; not an absolute dB threshold.",
             6.0f);
  spec.param(moving_average_window_, "moving_average_window", "Moving-average window",
             "Half-width (in frequency bins) of the CFAR training window on each side of the cell "
             "under test (moving_average mode).",
             64);
  spec.param(guard_bins_, "guard_bins", "Guard bins",
             "Number of guard bins skipped on each side of the cell under test so a wide signal "
             "does not contaminate its own noise estimate (moving_average mode).",
             4);
  spec.param(baseline_frames_, "baseline_frames", "Baseline frames",
             "Number of initial frames folded into the noise-floor baseline before detection "
             "begins (baseline mode).",
             16);
  spec.param(min_std_db_, "min_std_db", "Minimum std (dB)",
             "Floor applied to the estimated standard deviation to avoid divide-by-noise on flat "
             "regions.",
             0.5f);
  spec.param(emit_stride_, "emit_stride", "Emit stride",
             "Emit one detection mask every N processed frames.", 1);
}

void PowerDetection::initialize() {
  holoscan::Operator::initialize();
  if (burst_size_.get() <= 0 || num_bursts_.get() <= 0) {
    throw std::runtime_error("power_detection requires positive burst_size and num_bursts");
  }
  // Device buffers are allocated lazily on the first frame's stream so all work
  // (FFT, power, detection) shares the operator's incoming CUDA stream.
}

void PowerDetection::ensure_allocated(cudaStream_t /*stream*/) {
  if (state_.allocated) {
    return;
  }
  const size_t rows = static_cast<size_t>(num_bursts_.get());
  const size_t cols = static_cast<size_t>(burst_size_.get());
  const size_t cells = rows * cols;

  throw_if_cuda_error(cudaMalloc(&state_.fft_out, cells * sizeof(complex)), "cudaMalloc(fft_out)");
  throw_if_cuda_error(cudaMalloc(&state_.power_db, cells * sizeof(float)), "cudaMalloc(power_db)");
  throw_if_cuda_error(cudaMalloc(&state_.baseline_sum, cols * sizeof(double)),
                      "cudaMalloc(baseline_sum)");
  throw_if_cuda_error(cudaMalloc(&state_.baseline_sumsq, cols * sizeof(double)),
                      "cudaMalloc(baseline_sumsq)");
  throw_if_cuda_error(cudaMemset(state_.baseline_sum, 0, cols * sizeof(double)),
                      "cudaMemset(baseline_sum)");
  throw_if_cuda_error(cudaMemset(state_.baseline_sumsq, 0, cols * sizeof(double)),
                      "cudaMemset(baseline_sumsq)");

  int fft_dims[1] = {burst_size_.get()};
  throw_if_cufft_error(cufftPlanMany(&state_.fft_plan,
                                     1,
                                     fft_dims,
                                     fft_dims,
                                     1,
                                     burst_size_.get(),
                                     fft_dims,
                                     1,
                                     burst_size_.get(),
                                     CUFFT_C2C,
                                     num_bursts_.get()),
                       "cufftPlanMany");
  state_.fft_plan_initialized = true;
  state_.allocated = true;
}

void PowerDetection::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
  auto maybe_input = op_input.receive<power_detection_in_t>("in");
  if (!maybe_input) {
    return;
  }
  auto input = maybe_input.value();
  auto& in_tensor = std::get<0>(input);
  cudaStream_t stream = std::get<1>(input);

  const int rows = num_bursts_.get();
  const int cols = burst_size_.get();
  if (in_tensor.Size(0) < rows || in_tensor.Size(1) < cols) {
    HOLOSCAN_LOG_WARN("power_detection: input {}x{} smaller than configured {}x{}; skipping frame.",
                      in_tensor.Size(0), in_tensor.Size(1), rows, cols);
    return;
  }

  ensure_allocated(stream);

  auto meta = metadata();
  const uint16_t channel_number =
      meta ? meta->get<uint16_t>("channel_number", static_cast<uint16_t>(channel_filter_.get()))
           : static_cast<uint16_t>(channel_filter_.get());

  // 1) FFT of the raw IQ (this baseline owns its own transform).
  throw_if_cufft_error(cufftSetStream(state_.fft_plan, stream), "cufftSetStream");
  throw_if_cufft_error(cufftExecC2C(state_.fft_plan,
                                    reinterpret_cast<cufftComplex*>(in_tensor.Data()),
                                    reinterpret_cast<cufftComplex*>(state_.fft_out),
                                    CUFFT_FORWARD),
                       "cufftExecC2C");

  // 2) Power spectrogram in dB (fftshifted along frequency).
  const dim3 block2d(32, 8);
  const dim3 grid2d((cols + block2d.x - 1) / block2d.x, (rows + block2d.y - 1) / block2d.y);
  power_db_shift_kernel<<<grid2d, block2d, 0, stream>>>(state_.fft_out, state_.power_db, rows, cols);
  throw_if_cuda_error(cudaGetLastError(), "power_db_shift_kernel launch");

  ++frame_number_;
  const std::string mode = threshold_mode_.get();
  const bool baseline_mode = (mode == "baseline");

  // In baseline mode, fold this frame into the noise-floor estimate first.
  if (baseline_mode) {
    const int block1d = 256;
    const int grid1d = (cols + block1d - 1) / block1d;
    baseline_accumulate_kernel<<<grid1d, block1d, 0, stream>>>(
        state_.power_db, state_.baseline_sum, state_.baseline_sumsq, rows, cols);
    throw_if_cuda_error(cudaGetLastError(), "baseline_accumulate_kernel launch");
    state_.baseline_samples += static_cast<uint64_t>(rows);
  }

  // Respect emit stride (baseline stats above are still updated every frame).
  const int stride = std::max(1, emit_stride_.get());
  const bool warming_up = baseline_mode && frame_number_ <= static_cast<uint64_t>(baseline_frames_.get());
  const bool emit_this_frame = (frame_number_ % static_cast<uint64_t>(stride)) == 0 && !warming_up;
  if (!emit_this_frame) {
    if (meta) {
      meta->set("power_detection_emitted", false);
      meta->set("power_detection_warming_up", warming_up);
    }
    return;
  }

  // 3) Detection -> mask, written directly into the owned emit buffer.
  const size_t cells = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  auto mask_device = make_owned_device_u8(cells);

  if (baseline_mode) {
    const double nsamp = static_cast<double>(std::max<uint64_t>(1, state_.baseline_samples));
    baseline_detect_kernel<<<grid2d, block2d, 0, stream>>>(state_.power_db,
                                                            state_.baseline_sum,
                                                            state_.baseline_sumsq,
                                                            nsamp,
                                                            mask_device.get(),
                                                            rows,
                                                            cols,
                                                            zscore_threshold_.get(),
                                                            min_std_db_.get());
    throw_if_cuda_error(cudaGetLastError(), "baseline_detect_kernel launch");
  } else {
    cfar_zscore_kernel<<<grid2d, block2d, 0, stream>>>(state_.power_db,
                                                       mask_device.get(),
                                                       rows,
                                                       cols,
                                                       std::max(1, moving_average_window_.get()),
                                                       std::max(0, guard_bins_.get()),
                                                       zscore_threshold_.get(),
                                                       min_std_db_.get());
    throw_if_cuda_error(cudaGetLastError(), "cfar_zscore_kernel launch");
  }

  // Ensure the mask is fully computed before handing the buffer downstream.
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
    meta->set("power_detection_emitted", true);
    meta->set("power_detection_mode", mode);
    meta->set("power_detection_zscore_threshold", zscore_threshold_.get());
    meta->set("power_detection_frame_number", frame_number_);
  }
}

void PowerDetection::free_device_state() {
  if (state_.fft_plan_initialized) {
    cufftDestroy(state_.fft_plan);
    state_.fft_plan = 0;
    state_.fft_plan_initialized = false;
  }
  if (state_.fft_out) {
    cudaFree(state_.fft_out);
    state_.fft_out = nullptr;
  }
  if (state_.power_db) {
    cudaFree(state_.power_db);
    state_.power_db = nullptr;
  }
  if (state_.baseline_sum) {
    cudaFree(state_.baseline_sum);
    state_.baseline_sum = nullptr;
  }
  if (state_.baseline_sumsq) {
    cudaFree(state_.baseline_sumsq);
    state_.baseline_sumsq = nullptr;
  }
  state_.allocated = false;
}

void PowerDetection::stop() {
  HOLOSCAN_LOG_INFO("power_detection ch={} processed_frames={} emitted_masks={} mode={}",
                    channel_filter_.get(), frame_number_, detections_emitted_, threshold_mode_.get());
  free_device_state();
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

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
#include "finetuned_dino_mask_dump.hpp"
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <matx.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

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

// Predicate for the invalid-frame guard's occupancy count (thrust::count_if over the emitted mask).
struct ft_dino_is_nonzero {
  __host__ __device__ bool operator()(uint8_t v) const { return v != 0; }
};

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

// Fixed-bin dB histogram config for the robust-floor percentile estimate (covers any realistic
// gain-corrected dB range at 0.5 dB resolution; out-of-range pixels clamp to the end bins).
constexpr int   ft_hist_bins   = 320;
constexpr float ft_hist_min_db = -110.0f;
constexpr float ft_hist_max_db = 50.0f;

// Histogram of the (flattened, gain-corrected) dB image. Per-block shared histogram merged to a global
// one to bound global-atomic contention. Grid-stride over all pixels; launch shared = nbins * 4 bytes.
__global__ void ft_db_histogram_kernel(const float* __restrict__ db, long total, int nbins,
                                       float hist_min, float inv_bin_w, unsigned int* __restrict__ hist) {
  extern __shared__ unsigned int sh[];
  for (int i = threadIdx.x; i < nbins; i += blockDim.x) sh[i] = 0u;
  __syncthreads();
  const long stride = (long)blockDim.x * gridDim.x;
  for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x; idx < total; idx += stride) {
    int b = (int)((db[idx] - hist_min) * inv_bin_w);
    b = b < 0 ? 0 : (b >= nbins ? nbins - 1 : b);
    atomicAdd(&sh[b], 1u);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < nbins; i += blockDim.x)
    if (sh[i]) atomicAdd(&hist[i], sh[i]);
}

// From the histogram CDF, the dB values at low_pct and high_pct (linear interpolation within the bin).
// stats[0] = floor (low pct), stats[1] = high pct. Single thread (nbins is small, one serial scan).
__global__ void ft_hist_percentiles_kernel(const unsigned int* __restrict__ hist, int nbins,
                                           float hist_min, float bin_w, float low_pct, float high_pct,
                                           float* __restrict__ stats) {
  if (threadIdx.x != 0) return;
  long total = 0;
  for (int i = 0; i < nbins; ++i) total += hist[i];
  if (total <= 0) { stats[0] = hist_min; stats[1] = hist_min; return; }
  const double lo_target = 0.01 * (double)low_pct * (double)total;
  const double hi_target = 0.01 * (double)high_pct * (double)total;
  long cum = 0;
  float lo_db = hist_min, hi_db = hist_min + nbins * bin_w;
  bool lo_done = false, hi_done = false;
  for (int i = 0; i < nbins; ++i) {
    const long c = hist[i];
    if (!lo_done && (double)(cum + c) >= lo_target) {
      const double frac = (lo_target - (double)cum) / fmax((double)c, 1.0);
      lo_db = hist_min + ((float)i + (float)frac) * bin_w; lo_done = true;
    }
    if (!hi_done && (double)(cum + c) >= hi_target) {
      const double frac = (hi_target - (double)cum) / fmax((double)c, 1.0);
      hi_db = hist_min + ((float)i + (float)frac) * bin_w; hi_done = true;
    }
    cum += c;
    if (lo_done && hi_done) break;
  }
  stats[0] = lo_db; stats[1] = hi_db;
}

// out = in * w along the frequency axis (complex x real window); feeds the cuFFT plan in place.
// Same arithmetic as the previous matx expression `iq_view * w_view` (re*g, im*g).
__global__ void ft_window_rows_kernel(const float2* __restrict__ in, const float* __restrict__ w,
                                      float2* __restrict__ out, long total, int fft_size) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const float g = w[idx % fft_size];
  const float2 v = in[idx];
  out[idx] = make_float2(v.x * g, v.y * g);
}

// SHORT-FFT front-end (downsample_time_pool): fft_rows x fft_size spectrum -> fftshifted, gain-corrected
// dB on the reference row grid (img_rows x fft_size) by AREA averaging: output row r covers input rows
// [r*ratio, (r+1)*ratio) with fractional end weights (exact mean when ratio is integral). domain_db=1:
// mean of dB values (preserves a single periodogram's mean dB level, i.e. the trained noise level);
// 0: dB of the mean power (Bartlett average). Fused so the full-rate dB image is never materialized.
__global__ void ft_power_db_pool_kernel(const float2* __restrict__ spec, float* __restrict__ db_out,
                                        int fft_rows, int fft_size, int img_rows, float ratio,
                                        float gain_offset_db, int domain_db) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long total = (long)img_rows * fft_size;
  if (idx >= total) return;
  const int col = idx % fft_size;
  const int r = idx / fft_size;
  const int src_col = (col + fft_size / 2) % fft_size;
  const float a = r * ratio;
  const float b = (r + 1) * ratio;
  int i0 = static_cast<int>(floorf(a));
  int i1 = static_cast<int>(ceilf(b));
  if (i1 > fft_rows) i1 = fft_rows;
  float acc = 0.0f, wsum = 0.0f;
  for (int i = i0; i < i1; ++i) {
    const float w = fminf(b, (float)(i + 1)) - fmaxf(a, (float)i);
    if (w <= 0.0f) continue;
    const float2 c = spec[(long)i * fft_size + src_col];
    const float p = c.x * c.x + c.y * c.y + 1e-12f;
    acc += w * (domain_db ? 10.0f * log10f(p) : p);
    wsum += w;
  }
  const float m = acc / fmaxf(wsum, 1e-12f);
  db_out[idx] = (domain_db ? m : 10.0f * log10f(m)) - gain_offset_db;
}

// Normalization clip window decided ON THE GPU (no host round-trip): params = [clip_vmin, inv_span,
// use_robust]. Same arithmetic as the previous host code (double for the scalar terms).
__global__ void ft_set_clip_params_kernel(float* params, float vmin, float inv_span) {
  if (threadIdx.x || blockIdx.x) return;
  params[0] = vmin; params[1] = inv_span; params[2] = 0.0f;
}
__global__ void ft_clip_params_robust_kernel(const float* stats, float fixed_vmin, float fixed_inv_span,
                                             double span, double floor_frac, float min_range,
                                             float floor_below, float* params) {
  if (threadIdx.x || blockIdx.x) return;
  const float floor_db = stats[0];
  const float drange = stats[1] - stats[0];
  const bool too_flat = drange < min_range;
  const bool floor_implausible = floor_db < fixed_vmin - floor_below;
  const bool use_robust = !too_flat && !floor_implausible;
  params[0] = use_robust ? static_cast<float>(static_cast<double>(floor_db) - floor_frac * span) : fixed_vmin;
  params[1] = use_robust ? static_cast<float>(1.0 / span) : fixed_inv_span;
  params[2] = use_robust ? 1.0f : 0.0f;
}
__global__ void ft_clip_params_ref_kernel(const float* reference, double span, double floor_frac,
                                          float* params) {
  if (threadIdx.x || blockIdx.x) return;
  params[0] = static_cast<float>(static_cast<double>(reference[0]) - floor_frac * span);
  params[1] = static_cast<float>(1.0 / span);
  params[2] = 1.0f;
}
__global__ void ft_clip_normalize_dev_kernel(const float* __restrict__ db, float* __restrict__ out,
                                             long total, const float* __restrict__ params) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const float vmin = params[0];
  const float inv_span = params[1];
  float v = (db[idx] - vmin) * inv_span;
  out[idx] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// Compact (rows x ncols) model-resolution mask -> wide (rows x wide_cols) emitted grid, nearest along
// the columns with PyTorch's legacy-nearest index (src = min(floor(dst * (in/out)), in-1), float math)
// so the output is identical to the previous torch nearest upsample.
__global__ void ft_expand_cols_nearest_kernel(const uint8_t* __restrict__ compact, int rows, int ncols,
                                              uint8_t* __restrict__ wide, int wide_cols, float scale) {
  const long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long total = (long)rows * wide_cols;
  if (idx >= total) return;
  const int r = idx / wide_cols;
  const int c = idx - (long)r * wide_cols;
  int src = static_cast<int>(floorf(static_cast<float>(c) * scale));
  if (src > ncols - 1) src = ncols - 1;
  wide[idx] = compact[(long)r * ncols + src];
}

// Invalid-frame guard on the GPU: count set pixels, then zero the emitted mask if the occupancy is
// over the threshold -- the same decision as before (occ = nz / px > thresh, doubles), no host sync.
__global__ void ft_count_nonzero_kernel(const uint8_t* __restrict__ m, long n, unsigned long long* count) {
  __shared__ unsigned int block_count;
  if (threadIdx.x == 0) block_count = 0u;
  __syncthreads();
  unsigned int local = 0u;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)blockDim.x * gridDim.x) {
    local += (m[i] != 0) ? 1u : 0u;
  }
  atomicAdd(&block_count, local);
  __syncthreads();
  if (threadIdx.x == 0 && block_count) atomicAdd(count, static_cast<unsigned long long>(block_count));
}
__global__ void ft_zero_if_over_kernel(uint8_t* __restrict__ m, long n,
                                       const unsigned long long* __restrict__ count, double count_n,
                                       double thresh) {
  const double occ = static_cast<double>(*count) / count_n;
  if (!(occ > thresh)) return;
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)blockDim.x * gridDim.x) {
    m[i] = 0;
  }
}

// Pool-backed emit buffers (same contract as coherent_power's acquire_pooled_u8_buffer): the previous
// per-frame cudaMalloc + cudaFree-in-deleter cost ~1-2 ms per frame and cudaFree synchronizes the
// whole device. Consumers finish their GPU reads of DetectorMaskMessage::device_pixels before they
// release it (the viz and snipper copy/synchronize first), so recycling on release is safe.
std::mutex& ft_mask_pool_mutex() { static std::mutex m; return m; }
std::map<size_t, std::vector<void*>>& ft_mask_pool() { static std::map<size_t, std::vector<void*>> p; return p; }
std::shared_ptr<uint8_t> ft_acquire_pooled_mask(size_t bytes) {
  void* ptr = nullptr;
  {
    std::lock_guard<std::mutex> lock(ft_mask_pool_mutex());
    auto& free_list = ft_mask_pool()[bytes];
    if (!free_list.empty()) { ptr = free_list.back(); free_list.pop_back(); }
  }
  if (ptr == nullptr) {
    throw_if_cuda_error(cudaMalloc(&ptr, bytes), "malloc pooled emit mask");
  }
  return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr), [bytes](uint8_t* p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lock(ft_mask_pool_mutex());
    ft_mask_pool()[bytes].push_back(p);
  });
}

inline int ft_blocks(long total, int threads) { return static_cast<int>((total + threads - 1) / threads); }

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
  cudaFree(db_hist_device);
  cudaFree(robust_stats_device);
  cudaFree(clip_params_device);
  cudaFree(guard_count_device);
  cudaFree(tile_batch_device);
  cudaFree(logits_device);
  cudaFree(tile_mask_device);
  cudaFree(compact_mask_device);
  cudaFree(window_device);
  if (host_stats_pinned) cudaFreeHost(host_stats_pinned);
  if (host_guard_count_pinned) cudaFreeHost(host_guard_count_pinned);
  if (fft_plan_valid) cufftDestroy(static_cast<cufftHandle>(fft_plan));
  for (cudaEvent_t* ev : {&ev_total0, &ev_total1, &ev_native0, &ev_native1}) {
    if (*ev) cudaEventDestroy(*ev);
    *ev = nullptr;
  }
  spec_device = nullptr;
  normalized_device = nullptr;
  db_device = nullptr;
  col_stat_device = nullptr;
  col_smooth_device = nullptr;
  frontend_reference_device = nullptr;
  db_hist_device = nullptr;
  robust_stats_device = nullptr;
  clip_params_device = nullptr;
  guard_count_device = nullptr;
  host_stats_pinned = nullptr;
  host_guard_count_pinned = nullptr;
  tile_batch_device = nullptr;
  logits_device = nullptr;
  tile_mask_device = nullptr;
  compact_mask_device = nullptr;
  window_device = nullptr;
  window_len = 0;
  fft_plan = 0;
  fft_plan_valid = false;
  fft_rows = fft_size = img_rows = img_cols = nfft = tile_rows = batch = 0;
}

void FinetunedDinoDetector::ChannelBuffers::ensure(size_t new_fft_rows, size_t new_fft_size,
                                                   size_t new_img_rows, size_t new_img_cols,
                                                   size_t new_nfft, size_t new_tile_rows,
                                                   size_t new_batch) {
  // Geometry-independent pieces, created once: pinned readback slots and the per-frame events.
  if (host_stats_pinned == nullptr) {
    throw_if_cuda_error(cudaMallocHost(&host_stats_pinned, 8 * sizeof(float)), "pinned stats");
    std::fill_n(host_stats_pinned, 8, 0.0f);
  }
  if (host_guard_count_pinned == nullptr) {
    throw_if_cuda_error(cudaMallocHost(&host_guard_count_pinned, sizeof(unsigned long long)), "pinned guard");
    *host_guard_count_pinned = 0ull;
  }
  for (cudaEvent_t* ev : {&ev_total0, &ev_total1, &ev_native0, &ev_native1}) {
    if (*ev == nullptr) throw_if_cuda_error(cudaEventCreate(ev), "event create");
  }
  if (new_fft_rows == fft_rows && new_fft_size == fft_size && new_img_rows == img_rows &&
      new_img_cols == img_cols && new_nfft == nfft && new_tile_rows == tile_rows && new_batch == batch) {
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
  cudaFree(db_hist_device);
  cudaFree(robust_stats_device);
  cudaFree(clip_params_device);
  cudaFree(guard_count_device);
  cudaFree(tile_batch_device);
  cudaFree(logits_device);
  cudaFree(tile_mask_device);
  cudaFree(compact_mask_device);
  cudaFree(window_device);
  if (fft_plan_valid) cufftDestroy(static_cast<cufftHandle>(fft_plan));
  fft_plan_valid = false;

  const size_t spec_elems = new_fft_rows * new_fft_size;
  const size_t img_elems = new_img_rows * new_img_cols;
  const size_t tile_elems = new_batch * new_tile_rows * new_nfft;
  throw_if_cuda_error(cudaMalloc(&window_device, new_fft_size * sizeof(float)), "malloc window");
  window_len = 0;  // refilled in compute() for the current window type + fft_size
  throw_if_cuda_error(cudaMalloc(&spec_device, spec_elems * sizeof(ft_dino_complex)), "malloc spec");
  throw_if_cuda_error(cudaMalloc(&normalized_device, img_elems * sizeof(float)), "malloc normalized");
  // Flatten scratch: full dB image + per-frequency floor buffers + scalar reference.
  throw_if_cuda_error(cudaMalloc(&db_device, img_elems * sizeof(float)), "malloc db");
  throw_if_cuda_error(cudaMalloc(&col_stat_device, new_img_cols * sizeof(float)), "malloc col_stat");
  throw_if_cuda_error(cudaMalloc(&col_smooth_device, new_img_cols * sizeof(float)), "malloc col_smooth");
  throw_if_cuda_error(cudaMalloc(&frontend_reference_device, sizeof(float)), "malloc frontend_reference");
  throw_if_cuda_error(cudaMalloc(&db_hist_device, ft_hist_bins * sizeof(unsigned int)), "malloc db_hist");
  throw_if_cuda_error(cudaMalloc(&robust_stats_device, 2 * sizeof(float)), "malloc robust_stats");
  throw_if_cuda_error(cudaMalloc(&clip_params_device, 4 * sizeof(float)), "malloc clip_params");
  throw_if_cuda_error(cudaMalloc(&guard_count_device, sizeof(unsigned long long)), "malloc guard_count");
  throw_if_cuda_error(cudaMalloc(&tile_batch_device, tile_elems * sizeof(float)), "malloc tiles");
  throw_if_cuda_error(cudaMalloc(&logits_device, tile_elems * sizeof(float)), "malloc logits");
  throw_if_cuda_error(cudaMalloc(&tile_mask_device, tile_elems * sizeof(uint8_t)), "malloc tile_mask");
  throw_if_cuda_error(cudaMalloc(&compact_mask_device, new_img_rows * new_nfft * sizeof(uint8_t)),
                      "malloc compact_mask");
  // One batched C2C plan per geometry (replaces matx::fft, which materialized the windowed input into a
  // per-call cudaMallocAsync temporary and issued an extra ~64 MiB device copy every frame).
  cufftHandle plan = 0;
  int n[1] = {static_cast<int>(new_fft_size)};
  if (cufftPlanMany(&plan, 1, n, nullptr, 1, static_cast<int>(new_fft_size), nullptr, 1,
                    static_cast<int>(new_fft_size), CUFFT_C2C, static_cast<int>(new_fft_rows)) != CUFFT_SUCCESS) {
    throw std::runtime_error("[finetuned_dino_detector] cufftPlanMany failed");
  }
  fft_plan = static_cast<int>(plan);
  fft_plan_valid = true;
  fft_rows = new_fft_rows;
  fft_size = new_fft_size;
  img_rows = new_img_rows;
  img_cols = new_img_cols;
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
  spec.param(torch_freeze_, "torch_freeze", "Freeze TorchScript",
             "torch::jit::freeze the module at load: weights become constants and the per-forward fp32->bf16 "
             "weight casts of the autocast-traced exports are folded once (bit-identical logits verified on "
             "M5-S; ~1 ms per 2-tile forward). false = run the module as exported.", true);
  spec.param(real_time_downsample_, "real_time_downsample", "Downsample mode",
             "Wide FFT + bilinear resize freq to model width before inference (bounded cost). "
             "false = native per-tile path.", false);
  spec.param(downsample_fft_size_, "downsample_fft_size", "Downsample FFT size",
             "Wide FFT size for downsample mode (reproduces the app's dynamic spectrogram; e.g. 10240 "
             "at 245.76 MSps, 20480 at 500 MHz). Ignored when real_time_downsample=false.", 10240);
  spec.param(downsample_time_pool_, "downsample_time_pool", "Downsample time pooling",
             "Short-FFT front-end for the downsample path: FFT at downsample_fft_size (must be smaller than "
             "the auto/reference size) and area-average rows back onto the reference row grid "
             "(frame / auto_fft_size(rate) rows), so the model keeps its trained row time and 2 tiles per "
             "frame while each column integrates every bin (2:1 / 4:1 bilinear instead of 10:1 / 20:1). "
             "Requires flatten_noise_floor. Emitted mask grid is unchanged. Default off.", false);
  spec.param(downsample_time_pool_domain_, "downsample_time_pool_domain", "Downsample time pool domain",
             "db = mean of dB values (keeps a single periodogram's dB level, the trained noise level); "
             "power = dB of the mean power.", std::string("db"));
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
  // Default OFF is for LEGACY checkpoints (the old dino_fine_tuning frames_to_db path did no
  // flattening). The shipped M3_491 model is fine-tuned WITH the per-frequency floor flatten -- see
  // dino_fine_tuning/src/frontend.py (flatten=True; reference_q 75 / smooth 0.005 / max_boost 12 /
  // signal_cap 6, matching the operator defaults below) -- so the M3 live+loopback configs set this
  // TRUE. Do NOT turn it off for M3: inference would then diverge from the training front-end.
  spec.param(flatten_noise_floor_, "flatten_noise_floor", "Flatten noise floor",
             "Estimate a smooth per-frequency floor and lift low-floor bins up to a data-derived "
             "reference before inference. Default OFF for legacy models; the M3_491 configs set it true "
             "because M3 was fine-tuned WITH flatten -- keep it matched to the model's training.", false);
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
  spec.param(adaptive_normalization_, "adaptive_normalization", "Adaptive normalization",
             "Flatten path only: anchor the [0,1] clip to the per-frame data-derived floor (flatten "
             "reference) with adaptive_span_db instead of the fixed db_vmin/db_vmax window, so a narrow "
             "live scene fills [0,1] like training and moderate signals aren't compressed into the noise. "
             "false = validated fixed clip (default).", false);
  spec.param(adaptive_span_db_, "adaptive_span_db", "Adaptive span dB",
             "dB above the floor spread across [adaptive_floor_frac, 1] when adaptive_normalization is on. "
             "Smaller = more contrast/recall but more noise FP.", 34.0);
  spec.param(adaptive_floor_frac_, "adaptive_floor_frac", "Adaptive floor frac",
             "Where the per-frame noise floor lands in [0,1] under adaptive normalization.", 0.12);
  spec.param(adaptive_robust_floor_, "adaptive_robust_floor", "Adaptive robust floor",
             "Adaptive path: estimate the floor as a LOW percentile of the whole dB image (robust to "
             "occupancy) instead of the q-blend flatten reference, and fall back to the fixed calibrated "
             "clip when the frame has no usable dynamic range (dense/mostly-signal) or an implausibly low "
             "floor (noiseless). This is the density/distance-robust path. true = default.", true);
  spec.param(adaptive_low_pct_, "adaptive_low_pct", "Adaptive low percentile",
             "Noise-floor percentile of the dB image (0-100) for the robust floor.", 20.0);
  spec.param(adaptive_high_pct_, "adaptive_high_pct", "Adaptive high percentile",
             "High percentile of the dB image (0-100); (high-low) gauges the frame's dynamic range.", 95.0);
  spec.param(adaptive_min_range_db_, "adaptive_min_range_db", "Adaptive min range dB",
             "Fall back to the fixed clip when (high_pct - low_pct) < this (dB): the frame is "
             "mostly-signal/flat with no usable floor-to-signal spread.", 8.0);
  spec.param(adaptive_floor_below_calib_db_, "adaptive_floor_below_calib_db", "Adaptive floor-below-calib dB",
             "Fall back to the fixed clip when the robust floor is more than this many dB BELOW the fixed "
             "calibrated vmin: an implausibly low floor (noiseless/degenerate frame).", 25.0);
  spec.param(circular_edge_inference_, "circular_edge_inference", "Circular edge inference",
             "Downsample path: also run the model on a half-band circular rotation of the input "
             "(batched into the same forward) and take the mask's outer quarters from that pass, so "
             "no output column comes from the model's borders. Fixes the checkpoint's measured "
             "border bias (fires on flat noise at its image edges) without trimming detections; the "
             "DFT spectrum is periodic so the rotation is spectrally contiguous. Costs ~2x tiles per "
             "emitted frame.", true);
  spec.param(ignore_sideband_percent_, "ignore_sideband_percent", "Ignore sideband percent",
             "Zero the emitted mask's outermost columns on EACH band edge, as a percent of the mask "
             "width. The X410 anti-alias rolloff cliff at the extreme bins survives the floor flatten "
             "and fires the model there; this trims those columns after inference. Takes precedence "
             "over ignore_sideband_hz. 0 = off (same semantics as the coherent detector's).", 0.0);
  spec.param(ignore_sideband_hz_, "ignore_sideband_hz", "Ignore sideband Hz",
             "Alternative per-side trim as a frequency span (Hz), converted with the live rate. Used "
             "only when ignore_sideband_percent == 0. 0 = off.", 0.0);
  spec.param(debug_mask_dump_dir_, "debug_mask_dump_dir", "Debug mask dump dir",
             "VALIDATION ONLY: write each emitted mask as a .npy (+ manifest CSV) to this dir on a "
             "background thread, for the offline-vs-loopback A/B. \"\" = off (default; zero hot-path "
             "cost).", std::string(""));
  spec.param(debug_mask_dump_max_frames_, "debug_mask_dump_max_frames", "Debug mask dump max frames",
             "Total masks to write before the dump goes quiet. 0 = unlimited.", 0);
  spec.param(debug_spectrogram_dump_, "debug_spectrogram_dump", "Debug spectrogram dump",
             "When mask dump is on, also write the RT input spectrogram (float32, same grid) per frame "
             "for mask-on-spectrogram overlays. false = masks only (default).", false);
  spec.param(invalid_frame_guard_, "invalid_frame_guard", "Invalid-frame guard",
             "Suppress (emit empty mask) a frame whose occupancy is a gross outlier vs an adaptive "
             "baseline -- drop-corrupted frames under ingest saturation otherwise fire a spurious "
             "off-band blob. true = on (default).", true);
  spec.param(invalid_frame_min_occupancy_, "invalid_frame_min_occupancy", "Invalid-frame min occupancy",
             "Absolute occupancy floor (fraction 0-1): a mask denser than this is suppressed regardless "
             "of baseline. Set above the densest LEGITIMATE frame for the scene (OTA ~0.014; raise for "
             "dense composites).", 0.03);
  spec.param(invalid_frame_occupancy_k_, "invalid_frame_occupancy_k", "Invalid-frame occupancy k",
             "Also suppress when occupancy exceeds k * the adaptive baseline (catches outliers on denser "
             "scenes where the absolute floor is too high).", 6.0);
  spec.param(invalid_frame_baseline_alpha_, "invalid_frame_baseline_alpha", "Invalid-frame baseline alpha",
             "EWMA rate for the per-channel accepted-frame occupancy baseline (0-1).", 0.05);
}

void FinetunedDinoDetector::initialize() {
  holoscan::Operator::initialize();
  const int channels = channel_filter_.get() >= 0 ? 1 : std::max(1, num_channels_.get());
  frame_count_.assign(channels, 0);
  occ_baseline_.assign(channels, -1.0);   // <0 => not yet initialized (invalid-frame guard)
  prev_accept_fingerprint_.assign(channels, 0ull);
  guard_debug_ = [] { const char* e = std::getenv("DINO_GUARD_DEBUG"); return e && std::atoi(e) != 0; }();
  if (const char* e = std::getenv("DINO_DEBUG_STALL_FRAME"); e && e[0] != '\0') {
    debug_stall_frame_ = std::strtoull(e, nullptr, 10);
  }
  if (const char* e = std::getenv("DINO_DEBUG_STALL_MS"); e && e[0] != '\0') {
    debug_stall_ms_ = std::atoi(e);
  }
  if (debug_stall_frame_ != 0 && debug_stall_ms_ > 0) {
    std::fprintf(stderr, "[finetuned_dino_detector] DEBUG STALL armed: frame %llu will sleep %d ms (test only)\n",
                 static_cast<unsigned long long>(debug_stall_frame_), debug_stall_ms_);
  }
  channel_buffers_.assign(channels, ChannelBuffers{});
  runtime_ = std::make_shared<FinetunedDinoTorchRuntime>();
  if (!runtime_->load(model_script_path_.get(), torch_dtype_.get(), torch_freeze_.get())) {
    std::fprintf(stderr, "[finetuned_dino_detector] WARN: TorchScript not loaded at init\n");
  } else {
    // Prime the model so the first real frame doesn't pay the ~hundreds-of-ms cuDNN-autotune cost
    // (which would otherwise spike backpressure at live startup). Tiles are tile_rows x nfft; batch 2
    // covers the downsample path (512 rows) and primes the same per-tile kernels the native path uses.
    // Downsample-path forward is ~2 tiles, x2 when circular edge inference batches the rotated
    // copy -- prime the larger batch so the first live frame doesn't autotune.
    runtime_->warmup(tile_rows_.get(), nfft_.get(), 4);
    // Measured on the loopback stress test (2026-10-21): the batch-4 warmup alone still left the
    // FIRST live downsample frame at ~210 ms (batch-2 forward + the bilinear resize / nearest
    // upsample kernels had never run) and the native path at its full first-batch autotune. Warm
    // the EXACT live path for the common wide geometries (512 rows x 10240 @245.76 / 20480 @491.52)
    // or the native tile batch (5120 rows / 256 = 20 tiles @245.76; 40 @491.52 autotunes once more).
    if (real_time_downsample_.get()) {
      const int rows = 512;
      for (const int wide : {10240, 20480, 2048, 4096}) {
        const bool fixed = downsample_fft_size_.get() > 0;
        if (fixed && std::max(nfft_.get(), downsample_fft_size_.get()) != wide) continue;
        if (!fixed && wide < 10240) continue;
        float* dummy = nullptr;
        uint8_t* dummy_mask = nullptr;
        if (cudaMalloc(&dummy, static_cast<size_t>(rows) * wide * sizeof(float)) != cudaSuccess) continue;
        if (cudaMalloc(&dummy_mask, static_cast<size_t>(rows) * nfft_.get()) != cudaSuccess) { cudaFree(dummy); continue; }
        cudaMemset(dummy, 0, static_cast<size_t>(rows) * wide * sizeof(float));
        for (int it = 0; it < 2; ++it) {
          (void)runtime_->forward_downsampled(dummy, rows, wide, tile_rows_.get(), nfft_.get(),
                                              static_cast<float>(threshold_.get()), dummy_mask,
                                              nullptr, false, circular_edge_inference_.get());
        }
        cudaDeviceSynchronize();
        cudaFree(dummy);
        cudaFree(dummy_mask);
      }
      std::fprintf(stderr, "[finetuned_dino_detector] downsample-path warmup done\n");
    } else {
      runtime_->warmup(tile_rows_.get(), nfft_.get(), 20, 2);
    }
  }

  // Validation mask dump (inert unless a dir is configured). make_shared here (complete type) so
  // the deleter is type-erased for the header-only forward declaration in the app TUs.
  mask_dump_ = std::make_shared<MaskDumpWriter>();
  mask_dump_->configure(debug_mask_dump_dir_.get(), debug_mask_dump_max_frames_.get());
}

void FinetunedDinoDetector::stop() {
  if (mask_dump_) mask_dump_->flush_and_stop();
  release_channel_buffers();
}

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

  // Frame identity: prefer the CHDR converter's shared batch index (stamped on every emitted batch).
  // Keying the emitted mask to that same index -- the value the FFT stamps as fft_emitted_frame_number
  // and the snipper keys its IQ ring by -- makes the viz place masks correctly and the snipper find the
  // matching IQ, WITHOUT depending on the FFT operator and this IQ tap staying in lockstep (they drift
  // when the FFT lags/drops under load: its counter falls behind the true arrival index while this tap,
  // reading IQ directly, does not). Fall back to a local arrival counter offline (file replay has no
  // CHDR stamp), preserving the validated offline behavior. The local counter is advanced either way so
  // emit_stride cadence is identical across both paths.
  const uint64_t local_frame_number = ++frame_count_[ch];
  const uint64_t frame_number =
      (meta && meta->has_key("chdr_batch_index"))
          ? meta->get<uint64_t>("chdr_batch_index", local_frame_number)
          : local_frame_number;
  if (emit_stride_.get() > 1 && (frame_number % emit_stride_.get()) != 0) return;
  if (debug_stall_frame_ != 0 && debug_stall_ms_ > 0 && frame_number == debug_stall_frame_) {
    std::fprintf(stderr, "[finetuned_dino_detector] DEBUG STALL: frame %llu sleeping %d ms\n",
                 static_cast<unsigned long long>(frame_number), debug_stall_ms_);
    std::this_thread::sleep_for(std::chrono::milliseconds(debug_stall_ms_));
  }

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
  // Reference grid = the rows/cols of the EMITTED mask (and of the model image unless time pooling):
  // rows of ref_fft samples. Normally ref_fft == fft_size. With downsample_time_pool, the transform is
  // shorter (fft_size < ref_fft) and the dB image is area-pooled back onto the reference rows.
  int ref_fft = fft_size;
  bool pool = false;
  if (downsample && downsample_time_pool_.get()) {
    const int auto_ref = std::max(nfft, auto_fft_size(rate_hz));
    if (downsample_fft_size_.get() > 0 && fft_size < auto_ref && flatten_noise_floor_.get()) {
      pool = true;
      ref_fft = auto_ref;
    } else if (!time_pool_warned_) {
      std::fprintf(stderr, "[finetuned_dino_detector] WARN: downsample_time_pool ignored (needs "
                   "flatten_noise_floor and downsample_fft_size < auto size %d; got %d)\n", auto_ref, fft_size);
      time_pool_warned_ = true;
    }
  }

  // Total complex IQ samples in this frame -> whole transform rows and reference rows.
  const long total_samples = static_cast<long>(iq_tensor.Size(0)) * static_cast<long>(iq_tensor.Size(1));
  const int fft_rows = static_cast<int>(total_samples / fft_size);
  const int rows = static_cast<int>(total_samples / ref_fft);   // emitted mask rows == model image rows
  if (fft_rows <= 0 || rows <= 0) return;
  const int img_rows = pool ? rows : fft_rows;   // == rows when not pooling
  const int img_cols = fft_size;
  const int batch = (img_rows + tile_rows - 1) / tile_rows;

  auto& buf = channel_buffers_[ch];
  // Run all operator work on the IQ producer's stream so ordering with the upstream copy holds.
  const cudaStream_t stream = iq_stream;
  buf.ensure(static_cast<size_t>(fft_rows), static_cast<size_t>(fft_size), static_cast<size_t>(img_rows),
             static_cast<size_t>(img_cols), static_cast<size_t>(nfft), static_cast<size_t>(tile_rows),
             static_cast<size_t>(batch));
  // Whole-operator GPU span for this frame (FFT + front-end + model + guard) between two persistent
  // events; read after the single per-frame sync below.
  cudaEventRecord(buf.ev_total0, stream);

  // ---- front-end: FFT (native nfft, or wide for downsample) -> dB -> [0,1] normalized ------
  auto* iq_ptr = reinterpret_cast<float2*>(iq_tensor.Data());
  auto* spec_f2_mut = reinterpret_cast<float2*>(buf.spec_device);
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
    // the host vector dies at scope exit: make sure the (pageable) copy has consumed it
    throw_if_cuda_error(cudaStreamSynchronize(stream), "window H2D sync");
    buf.window_len = static_cast<size_t>(fft_size);
  }
  // Per-row (batched) 1D FFT over the frequency axis via one cached cuFFT plan (unnormalized, like the
  // previous matx::fft / torch). Windowed: window kernel writes the spectrum buffer, FFT in place.
  const int threads = 256;
  const long spec_total = static_cast<long>(fft_rows) * fft_size;
  const cufftHandle plan = static_cast<cufftHandle>(buf.fft_plan);
  if (cufftSetStream(plan, stream) != CUFFT_SUCCESS) throw std::runtime_error("[finetuned_dino_detector] cufftSetStream");
  if (use_window) {
    ft_window_rows_kernel<<<ft_blocks(spec_total, threads), threads, 0, stream>>>(
        iq_ptr, buf.window_device, spec_f2_mut, spec_total, fft_size);
    if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(spec_f2_mut), reinterpret_cast<cufftComplex*>(spec_f2_mut),
                     CUFFT_FORWARD) != CUFFT_SUCCESS) {
      throw std::runtime_error("[finetuned_dino_detector] cufftExecC2C failed");
    }
  } else {
    if (cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(iq_ptr), reinterpret_cast<cufftComplex*>(spec_f2_mut),
                     CUFFT_FORWARD) != CUFFT_SUCCESS) {
      throw std::runtime_error("[finetuned_dino_detector] cufftExecC2C failed");
    }
  }

  // Processing-gain correction: an unnormalized N-pt FFT scales a white-noise bin's power by N, so a
  // wide (downsample) FFT sits 10*log10(fft_size/nfft) dB hotter than the nfft-pt scale the model was
  // trained on. Shift vmin up by that offset (== subtract it from dB) to re-center the normalized image
  // onto the trained intensity distribution -- otherwise low-SNR signal near the noise floor is lost.
  // Zero for native mode (fft_size == nfft); derived from FFT sizes only, so it reproduces on any OTA
  // rate/geometry with no per-capture calibration.
  const double gain_offset_db = 10.0 * std::log10(static_cast<double>(fft_size) / static_cast<double>(nfft));
  // Training-power level match: see match_training_power_level / power_level_trim_db params.
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
    if (pool) {
      std::fprintf(stderr, "[finetuned_dino_detector] short-FFT front-end: %d-pt FFT x %d rows -> area-pooled "
                   "(%s) to %d rows (%.2f rows each) on the %d-pt reference grid\n", fft_size, fft_rows,
                   downsample_time_pool_domain_.get().c_str(), rows,
                   static_cast<double>(fft_rows) / rows, ref_fft);
    }
    level_log_emitted_ = true;
  }
  const float inv_span = 1.0f / std::max(static_cast<float>(db_vmax_.get() - db_vmin_.get()), 1e-6f);
  const long img_total = static_cast<long>(img_rows) * img_cols;
  const int img_blocks = ft_blocks(img_total, threads);
  const auto* spec_f2 = reinterpret_cast<const float2*>(buf.spec_device);
  bool stage_frontend_logs = false;
  float flatten_sigma = 0.0f;

  if (flatten_noise_floor_.get()) {
    // Split power->dB->clip so the per-frequency floor can be flattened on the dB image first.
    // gain_offset is subtracted here (trained-scale dB), so the clip below uses the raw db_vmin.
    if (pool) {
      ft_power_db_pool_kernel<<<img_blocks, threads, 0, stream>>>(
          spec_f2, buf.db_device, fft_rows, fft_size, img_rows,
          static_cast<float>(static_cast<double>(fft_rows) / static_cast<double>(img_rows)),
          static_cast<float>(gain_offset_db), downsample_time_pool_domain_.get() == "power" ? 0 : 1);
    } else {
      ft_power_to_db_shift_kernel<<<img_blocks, threads, 0, stream>>>(
          spec_f2, buf.db_device, img_rows, img_cols, static_cast<float>(gain_offset_db));
    }
    // Bandwidth-invariant smoothing: sigma tracks a fixed fraction of the band, so the same config
    // works at any sample rate (the filter rolloff occupies a fixed fraction of Nyquist).
    const float sigma = std::max(2.0f, static_cast<float>(flatten_smooth_frac_.get()) * img_cols);
    flatten_sigma = sigma;
    const int smooth_radius = std::max(1, static_cast<int>(std::ceil(sigma * 1.5f)));
    const int col_blocks = (img_cols + threads - 1) / threads;
    const float q = static_cast<float>(flatten_reference_q_.get() / 100.0);
    // Pass 1: uncapped per-freq floor -> smooth -> reference. Pass 2 (if signal-cap on): recompute the
    // floor with signal influence capped at reference+cap so strong bins don't inflate their own floor.
    ft_col_mean_kernel<<<img_cols, threads, 0, stream>>>(buf.db_device, img_rows, img_cols, nullptr, 0.0f,
                                                         buf.col_stat_device);
    ft_smooth_cols_kernel<<<col_blocks, threads, 0, stream>>>(buf.col_stat_device, img_cols, smooth_radius,
                                                              sigma, buf.col_smooth_device);
    ft_frontend_reference_kernel<<<1, threads, 0, stream>>>(buf.col_smooth_device, img_cols, q,
                                                            buf.frontend_reference_device);
    if (flatten_signal_cap_db_.get() > 0.0) {
      ft_col_mean_kernel<<<img_cols, threads, 0, stream>>>(
          buf.db_device, img_rows, img_cols, buf.frontend_reference_device,
          static_cast<float>(flatten_signal_cap_db_.get()), buf.col_stat_device);
      ft_smooth_cols_kernel<<<col_blocks, threads, 0, stream>>>(buf.col_stat_device, img_cols, smooth_radius,
                                                                sigma, buf.col_smooth_device);
      ft_frontend_reference_kernel<<<1, threads, 0, stream>>>(buf.col_smooth_device, img_cols, q,
                                                              buf.frontend_reference_device);
    }
    ft_frontend_correction_kernel<<<img_blocks, threads, 0, stream>>>(
        buf.db_device, img_rows, img_cols, buf.col_smooth_device, buf.frontend_reference_device,
        static_cast<float>(flatten_max_boost_db_.get()));
    // Clip window: fixed calibrated, OR per-frame adaptive anchored to the data-derived floor -- decided
    // on the GPU (same arithmetic as before, no D2H round trip / stream sync).
    const float fixed_vmin = static_cast<float>(db_vmin_.get() + level_offset_db);  // gain already subtracted above
    const double span = std::max(1.0, adaptive_span_db_.get());
    if (adaptive_normalization_.get() && adaptive_robust_floor_.get()) {
      // Robust floor: a LOW percentile of the whole (flattened) dB image survives any occupancy, and a
      // small floor-to-signal spread OR an implausibly low floor triggers a fallback to the fixed clip
      // (dense/mostly-signal or noiseless frames, where a per-frame anchor collapses). See
      // notes/dino_ft_finetune_plan.md (robustness across field distance/density).
      constexpr int kBins = ft_hist_bins;
      const float bin_w = (ft_hist_max_db - ft_hist_min_db) / static_cast<float>(kBins);
      throw_if_cuda_error(cudaMemsetAsync(buf.db_hist_device, 0, kBins * sizeof(unsigned int), stream),
                          "robust hist memset");
      const int hist_blocks = std::min(img_blocks, 512);
      ft_db_histogram_kernel<<<hist_blocks, threads, kBins * sizeof(unsigned int), stream>>>(
          buf.db_device, img_total, kBins, ft_hist_min_db, 1.0f / bin_w, buf.db_hist_device);
      ft_hist_percentiles_kernel<<<1, 32, 0, stream>>>(
          buf.db_hist_device, kBins, ft_hist_min_db, bin_w,
          static_cast<float>(adaptive_low_pct_.get()), static_cast<float>(adaptive_high_pct_.get()),
          buf.robust_stats_device);
      ft_clip_params_robust_kernel<<<1, 1, 0, stream>>>(
          buf.robust_stats_device, fixed_vmin, inv_span, span, adaptive_floor_frac_.get(),
          static_cast<float>(adaptive_min_range_db_.get()),
          static_cast<float>(adaptive_floor_below_calib_db_.get()), buf.clip_params_device);
    } else if (adaptive_normalization_.get()) {
      ft_clip_params_ref_kernel<<<1, 1, 0, stream>>>(buf.frontend_reference_device, span,
                                                     adaptive_floor_frac_.get(), buf.clip_params_device);
    } else {
      ft_set_clip_params_kernel<<<1, 1, 0, stream>>>(buf.clip_params_device, fixed_vmin, inv_span);
    }
    ft_clip_normalize_dev_kernel<<<img_blocks, threads, 0, stream>>>(
        buf.db_device, buf.normalized_device, img_total, buf.clip_params_device);
    throw_if_cuda_error(cudaGetLastError(), "flatten frontend kernels");
    // One-time startup log values: stage them into pinned memory, print after the frame sync.
    if (!flatten_log_emitted_ || (adaptive_normalization_.get() && !robust_log_emitted_ && !adaptive_log_emitted_)) {
      cudaMemcpyAsync(buf.host_stats_pinned, buf.robust_stats_device, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(buf.host_stats_pinned + 2, buf.clip_params_device, 3 * sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(buf.host_stats_pinned + 5, buf.frontend_reference_device, sizeof(float), cudaMemcpyDeviceToHost, stream);
      stage_frontend_logs = true;
    }
  } else {
    // Original fused path (validated): fold the gain + power-level offsets into vmin, single kernel.
    const float vmin = static_cast<float>(db_vmin_.get() + vmin_offset_db);
    power_db_normalize_kernel<<<img_blocks, threads, 0, stream>>>(
        spec_f2, buf.normalized_device, img_rows, img_cols, vmin, inv_span);
    throw_if_cuda_error(cudaGetLastError(), "power_db_normalize kernel");
  }

  if (!(runtime_ && runtime_->loaded())) {
    if (!startup_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] inference unavailable (model not loaded)\n");
      startup_log_emitted_ = true;
    }
    return;  // don't emit a bogus mask
  }

  // Filled by whichever path runs; emitted once at the end. width = mask columns on the emitted grid.
  std::shared_ptr<uint8_t> emit_holder;
  int mask_width = 0;
  // Emitted mask = compact model-resolution mask with each column repeated col_repeat times (downsample
  // path): consumers can work on the compact form (snipper) with identical results.
  int col_repeat = 1;

  if (downsample) {
    // ---- downsample path: resize freq -> model width, tile, infer, threshold (torch) -> compact mask ----
    const bool ok = runtime_->forward_downsampled(buf.normalized_device, img_rows, img_cols, tile_rows, nfft,
                                                  static_cast<float>(threshold_.get()),
                                                  buf.compact_mask_device, stream, /*time_it=*/true,
                                                  circular_edge_inference_.get());
    if (!ok) return;
    mask_width = ref_fft;   // emitted on the wide (reference / native display) grid, as before
    emit_holder = ft_acquire_pooled_mask(static_cast<size_t>(rows) * mask_width);
    const long out_total = static_cast<long>(rows) * mask_width;
    ft_expand_cols_nearest_kernel<<<ft_blocks(out_total, threads), threads, 0, stream>>>(
        buf.compact_mask_device, rows, nfft, emit_holder.get(), mask_width,
        static_cast<float>(nfft) / static_cast<float>(mask_width));
    throw_if_cuda_error(cudaGetLastError(), "expand mask kernel");
    col_repeat = (mask_width % nfft == 0) ? mask_width / nfft : 1;
  } else {
    // ---- native path: tile into [B,1,tile_rows,nfft] (pad the last tile with zeros) -------------
    cudaEventRecord(buf.ev_native0, stream);
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
    const bool ok = runtime_->forward(buf.tile_batch_device, batch, tile_rows, nfft,
                                      buf.logits_device, stream);
    if (!ok) return;  // don't emit a bogus mask
    // ---- threshold + stitch tiles back to a native (rows x nfft) mask -------------------------
    const long tile_total = static_cast<long>(batch) * tile_stride;
    sigmoid_threshold_kernel<<<ft_blocks(tile_total, threads), threads, 0, stream>>>(
        buf.logits_device, buf.tile_mask_device, tile_total, static_cast<float>(threshold_.get()));
    throw_if_cuda_error(cudaGetLastError(), "sigmoid_threshold kernel");
    emit_holder = ft_acquire_pooled_mask(static_cast<size_t>(rows) * nfft);
    const size_t mask_row_bytes = static_cast<size_t>(nfft) * sizeof(uint8_t);
    for (int b = 0; b < batch; ++b) {
      const int r0 = b * tile_rows;
      const int valid = std::min(tile_rows, rows - r0);
      if (valid <= 0) break;
      throw_if_cuda_error(
          cudaMemcpyAsync(emit_holder.get() + static_cast<size_t>(r0) * nfft,
                          buf.tile_mask_device + static_cast<size_t>(b) * tile_stride,
                          static_cast<size_t>(valid) * mask_row_bytes, cudaMemcpyDeviceToDevice, stream),
          "stitch copy");
    }
    cudaEventRecord(buf.ev_native1, stream);
    mask_width = nfft;
  }
  uint8_t* emit_mask = emit_holder.get();

  // Band-edge sideband ignore (both paths): zero the outer mask columns per side. percent wins over
  // hz (converted with the live rate); capped at half the width. Runs after inference so the model
  // input is untouched -- only the emitted detections are trimmed.
  int ignore_cols = 0;
  {
    const double pct = ignore_sideband_percent_.get();
    const double hz = ignore_sideband_hz_.get();
    if (pct > 0.0) {
      ignore_cols = static_cast<int>(std::floor(static_cast<double>(mask_width) * pct / 100.0));
    } else if (hz > 0.0 && rate_hz > 0.0) {
      ignore_cols = static_cast<int>(std::ceil(hz / (rate_hz / static_cast<double>(mask_width))));
    }
    ignore_cols = std::min(ignore_cols, mask_width / 2);
    if (ignore_cols > 0) {
      throw_if_cuda_error(cudaMemset2DAsync(emit_mask, mask_width, 0, ignore_cols, rows, stream),
                          "sideband memset left");
      throw_if_cuda_error(cudaMemset2DAsync(emit_mask + (mask_width - ignore_cols), mask_width, 0,
                                            ignore_cols, rows, stream),
                          "sideband memset right");
      col_repeat = 1;   // the emitted mask is no longer an exact column replication of the compact one
    }
  }

  // INVALID-FRAME GUARD. Under real-time ingest saturation the pipeline drops packets, corrupting the
  // frame fill and making the segmenter fire a large spurious off-band blob (root cause confirmed in the
  // offline-vs-loopback A/B). A drop-corrupted frame is invalid, so the correct output is NO detections:
  // suppress (zero) the mask when its occupancy is a gross outlier vs an adaptive per-channel baseline.
  // The absolute floor catches spikes before the baseline warms up / on sparse scenes; the k*baseline
  // term protects genuinely dense scenes. Baseline updates only on ACCEPTED frames. Counting and the
  // conditional zeroing run on the GPU; the count is read back after the frame's one sync below.
  double guard_thresh = -1.0;
  double guard_count_n = 0.0;
  if (invalid_frame_guard_.get() && rows > 0 && mask_width > 0) {
    const double base = occ_baseline_[ch];
    guard_thresh = std::max(
        invalid_frame_min_occupancy_.get(),
        base >= 0.0 ? invalid_frame_occupancy_k_.get() * base : invalid_frame_min_occupancy_.get());
    // Exact replication -> count the compact mask (col_repeat x fewer pixels, same occupancy).
    const bool count_compact = downsample && col_repeat > 1;
    const uint8_t* count_src = count_compact ? buf.compact_mask_device : emit_mask;
    const long count_px = count_compact ? static_cast<long>(rows) * nfft : static_cast<long>(rows) * mask_width;
    const long emit_px = static_cast<long>(rows) * mask_width;
    guard_count_n = static_cast<double>(count_px);
    throw_if_cuda_error(cudaMemsetAsync(buf.guard_count_device, 0, sizeof(unsigned long long), stream),
                        "guard count memset");
    ft_count_nonzero_kernel<<<std::min(1024, ft_blocks(count_px, threads)), threads, 0, stream>>>(
        count_src, count_px, buf.guard_count_device);
    ft_zero_if_over_kernel<<<std::min(1024, ft_blocks(emit_px, threads)), threads, 0, stream>>>(
        emit_mask, emit_px, buf.guard_count_device, guard_count_n, guard_thresh);
    throw_if_cuda_error(cudaGetLastError(), "guard kernels");
    throw_if_cuda_error(cudaMemcpyAsync(buf.host_guard_count_pinned, buf.guard_count_device,
                                        sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream),
                        "guard count D2H");
  }

  // ---- the ONE per-frame synchronization: the mask must be complete before it is emitted ----------
  cudaEventRecord(buf.ev_total1, stream);
  throw_if_cuda_error(cudaEventSynchronize(buf.ev_total1), "frame sync");

  double detector_total_ms = -1.0;
  {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, buf.ev_total0, buf.ev_total1) == cudaSuccess) detector_total_ms = ms;
  }
  double detector_ms = -1.0;
  if (downsample) {
    detector_ms = runtime_->last_inference_ms();
  } else {
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, buf.ev_native0, buf.ev_native1) == cudaSuccess) detector_ms = ms;
  }
  if (detector_ms >= 0.0) {
    ++inference_samples_;
    inference_ms_ewma_ += (detector_ms - inference_ms_ewma_) / static_cast<double>(inference_samples_);
    if (inference_samples_ == 1 || (inference_samples_ % 32) == 0) {
      if (downsample) {
        std::fprintf(stderr, "[finetuned_dino_detector] downsample: FFT %d -> model %d (gain corr "
                     "%.2f dB), %d rows -> %d tiles; inference %.2f ms (mean %.2f ms over %llu frames)\n",
                     fft_size, nfft, gain_offset_db, img_rows, batch, detector_ms, inference_ms_ewma_,
                     static_cast<unsigned long long>(inference_samples_));
      } else {
        std::fprintf(stderr, "[finetuned_dino_detector] native: FFT %d (model width), %d rows -> %d tiles; "
                     "inference %.2f ms (mean %.2f ms over %llu frames)\n",
                     nfft, rows, batch, detector_ms, inference_ms_ewma_,
                     static_cast<unsigned long long>(inference_samples_));
      }
      std::fprintf(stderr, "[finetuned_dino_detector] frame %llu: total GPU %.2f ms (front-end + model path %.2f ms)\n",
                   static_cast<unsigned long long>(frame_number), detector_total_ms, detector_ms);
    }
  }

  // One-time startup logs (values staged into pinned memory before the sync).
  if (stage_frontend_logs) {
    const float* hs = buf.host_stats_pinned;
    if (adaptive_normalization_.get() && adaptive_robust_floor_.get() && !robust_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] robust normalization ON: floor(p%.0f)=%.2f dB "
                   "high(p%.0f)=%.2f dB range=%.2f dB -> %s (fixed_vmin=%.2f, clip_vmin=%.2f, span=%.1f)\n",
                   adaptive_low_pct_.get(), hs[0], adaptive_high_pct_.get(), hs[1], hs[1] - hs[0],
                   hs[4] > 0.5f ? "ROBUST" : "FALLBACK-FIXED",
                   static_cast<float>(db_vmin_.get() + level_offset_db), hs[2],
                   std::max(1.0, adaptive_span_db_.get()));
      robust_log_emitted_ = true;
    } else if (adaptive_normalization_.get() && !adaptive_robust_floor_.get() && !adaptive_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] adaptive normalization ON: per-frame floor=%.2f dB "
                   "span=%.1f dB floor_frac=%.2f -> clip_vmin=%.2f\n", hs[5],
                   std::max(1.0, adaptive_span_db_.get()), adaptive_floor_frac_.get(), hs[2]);
      adaptive_log_emitted_ = true;
    }
    if (!flatten_log_emitted_) {
      std::fprintf(stderr, "[finetuned_dino_detector] per-freq floor flatten ON (q=%.0f frac=%.4f "
                   "-> sigma=%.1f bins @ fft=%d, max_boost=%.1f dB signal_cap=%.1f dB)\n",
                   flatten_reference_q_.get(), flatten_smooth_frac_.get(), flatten_sigma, img_cols,
                   flatten_max_boost_db_.get(), flatten_signal_cap_db_.get());
      flatten_log_emitted_ = true;
    }
  }

  // Invalid-frame guard bookkeeping (the zeroing already happened on the GPU).
  if (guard_thresh >= 0.0) {
    const double occ = static_cast<double>(*buf.host_guard_count_pinned) / guard_count_n;
    double& base = occ_baseline_[ch];
    if (occ > guard_thresh) {
      if ((invalid_frames_suppressed_++ % 8) == 0) {
        std::fprintf(stderr, "[finetuned_dino_detector] invalid-frame guard: suppressed ch%d frame %llu "
                     "occ=%.2f%% > thresh=%.2f%% (baseline=%.3f%%); total suppressed=%llu\n",
                     static_cast<int>(channel_number), static_cast<unsigned long long>(frame_number),
                     100.0 * occ, 100.0 * guard_thresh, base >= 0.0 ? 100.0 * base : -1.0,
                     static_cast<unsigned long long>(invalid_frames_suppressed_));
      }
      // DIAGNOSTIC: characterize the corrupt frame from the ingest metadata (env DINO_GUARD_DEBUG).
      if (guard_debug_) {
        const uint32_t pib  = meta ? meta->get<uint32_t>("chdr_packets_in_batch", 0u) : 0u;
        const uint32_t pexp = meta ? meta->get<uint32_t>("chdr_expected_packets_in_batch", 0u) : 0u;
        const bool part     = meta ? meta->get<bool>("chdr_partial_batch", false) : false;
        const uint64_t fp   = meta ? meta->get<uint64_t>("chdr_content_fingerprint", 0ull) : 0ull;
        const uint64_t pfp  = (ch < prev_accept_fingerprint_.size()) ? prev_accept_fingerprint_[ch] : 0ull;
        std::fprintf(stderr,
            "[dino_guard_dbg] SUPPRESS ch%d frame=%llu occ=%.3f%% thr=%.3f%% base=%.4f%% "
            "pkts=%u/%u short=%d partial=%d fp=%016llx prev_fp=%016llx fp_eq_prev=%d\n",
            static_cast<int>(channel_number), static_cast<unsigned long long>(frame_number),
            100.0 * occ, 100.0 * guard_thresh, base >= 0.0 ? 100.0 * base : -1.0,
            pib, pexp, (pexp > 0u && pib < pexp) ? 1 : 0, part ? 1 : 0,
            static_cast<unsigned long long>(fp), static_cast<unsigned long long>(pfp),
            (fp != 0ull && fp == pfp) ? 1 : 0);
      }
    } else {
      base = (base < 0.0) ? occ
                          : (1.0 - invalid_frame_baseline_alpha_.get()) * base
                                + invalid_frame_baseline_alpha_.get() * occ;
      if (guard_debug_ && ch < prev_accept_fingerprint_.size()) {
        prev_accept_fingerprint_[ch] = meta ? meta->get<uint64_t>("chdr_content_fingerprint", 0ull) : 0ull;
      }
    }
  }

  // VALIDATION ONLY: copy the finalized device mask to host and hand it to the background writer,
  // so a real-time loopback run can be diffed against the deterministic offline eval. The D2H +
  // sync run only when the dump is configured and under budget -- zero cost on the normal path.
  if (mask_dump_ && mask_dump_->wants_more()) {
    const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(mask_width);
    if (dump_host_buf_.size() < n) dump_host_buf_.resize(n);
    throw_if_cuda_error(cudaMemcpyAsync(dump_host_buf_.data(), emit_mask, n * sizeof(uint8_t),
                                        cudaMemcpyDeviceToHost, stream), "mask dump D2H");
    // Optionally also grab the RT input spectrogram (normalized [0,1] image the model saw).
    const float* spec_host = nullptr;
    const size_t spec_n = static_cast<size_t>(img_rows) * static_cast<size_t>(img_cols);
    if (debug_spectrogram_dump_.get() && buf.normalized_device != nullptr && spec_n == n) {
      if (dump_spec_host_buf_.size() < n) dump_spec_host_buf_.resize(n);
      throw_if_cuda_error(cudaMemcpyAsync(dump_spec_host_buf_.data(), buf.normalized_device,
                                          n * sizeof(float), cudaMemcpyDeviceToHost, stream),
                          "spec dump D2H");
      spec_host = dump_spec_host_buf_.data();
    }
    throw_if_cuda_error(cudaStreamSynchronize(stream), "mask dump sync");
    MaskDumpFrameInfo info;
    info.host_mono_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    info.detector_ms = detector_total_ms;
    if (meta) {
      info.chdr_batch_index = meta->get<uint64_t>("chdr_batch_index", 0ull);
      info.chdr_pkts = meta->get<uint32_t>("chdr_packets_in_batch", 0u);
      info.chdr_expected_pkts = meta->get<uint32_t>("chdr_expected_packets_in_batch", 0u);
      info.chdr_partial = meta->get<bool>("chdr_partial_batch", false);
    }
    mask_dump_->submit(static_cast<int>(channel_number), frame_number, rows, mask_width,
                       dump_host_buf_.data(), spec_host, &info);
  }

  holoscan::ops::DetectorMaskMessage mask_msg;
  mask_msg.width = mask_width;
  mask_msg.height = rows;
  mask_msg.channel = channel_number;
  mask_msg.frame_number = frame_number;
  mask_msg.col_repeat = col_repeat;
  mask_msg.device_pixels = std::move(emit_holder);
  if (meta) {
    mask_msg.file_offset_complex    = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
    mask_msg.data_end_complex       = meta->get<uint64_t>("offline_source_data_end_complex", 0);
    mask_msg.frame_end_complex      = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
    mask_msg.complex_samples_read   = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
    mask_msg.complex_samples_padded = meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
  }
  if (meta) {
    meta->set("detector_inference_ms", detector_total_ms);   // whole-operator GPU time (must fit the frame period)
    meta->set("detector_model_path_ms", detector_ms);        // resize+forward+stitch (downsample) / tiles+forward+stitch (native)
  }
  op_output.emit(mask_msg, "mask_out");
  if (meta) meta->set("finetuned_dino_mask_emitted", true);
  ++compute_count_;
}

}  // namespace holoscan::ops

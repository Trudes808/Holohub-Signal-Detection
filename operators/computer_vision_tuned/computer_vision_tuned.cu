// SPDX-FileCopyrightText: 2026 The University of Texas at Austin
//
// SPDX-License-Identifier: Apache-2.0
#include "computer_vision_tuned.hpp"

// Shared detector output contract (see cuda_dino_detector / computer_vision_baseline).
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace holoscan::ops {

namespace {

constexpr float kPowerEps = 1e-20f;
constexpr int kBlockDim = 16;       // 2D kernels
constexpr int kReduceThreads = 256;

void throw_if_cuda_error(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("computer_vision_tuned CUDA error during ") + what + ": " +
                             cudaGetErrorString(status));
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

// Complex spectrogram -> dB magnitude image, fftshifted along frequency.
__global__ void complex_to_db_shift_kernel(const complex* __restrict__ in,
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
  out_db[static_cast<size_t>(row) * cols + col] = 10.0f * log10f(power);
}

// Per-frequency-column background estimate (sigma-clipped mean/std over time
// rows). One thread per column. A first pass gets the raw mean/std, a second
// pass re-estimates using only cells below mean + clip_z*std so that occupied
// (signal) rows do not inflate the floor. This is the local/adaptive background
// that replaces the naive global image threshold.
__global__ void column_background_kernel(const float* __restrict__ db,
                                         float* __restrict__ col_mean,
                                         float* __restrict__ col_std,
                                         int rows,
                                         int cols,
                                         float clip_z,
                                         float min_std) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols) {
    return;
  }
  double s = 0.0, ss = 0.0;
  for (int r = 0; r < rows; ++r) {
    const double x = db[static_cast<size_t>(r) * cols + col];
    s += x;
    ss += x * x;
  }
  double dn = static_cast<double>(rows);
  double mean = s / dn;
  double var = ss / dn - mean * mean;
  double std = sqrt(var > 0.0 ? var : 0.0);

  // One sigma-clip refinement pass (reject high-side outliers = signal).
  const double cut = mean + static_cast<double>(clip_z) * (std > min_std ? std : min_std);
  double s2 = 0.0, ss2 = 0.0;
  int n2 = 0;
  for (int r = 0; r < rows; ++r) {
    const double x = db[static_cast<size_t>(r) * cols + col];
    if (x <= cut) {
      s2 += x;
      ss2 += x * x;
      ++n2;
    }
  }
  if (n2 >= 2) {
    mean = s2 / n2;
    var = ss2 / n2 - mean * mean;
    std = sqrt(var > 0.0 ? var : 0.0);
  }
  col_mean[col] = static_cast<float>(mean);
  col_std[col] = static_cast<float>(std > min_std ? std : min_std);
}

// Local z-score thresholding into a high (seed) and low (candidate) mask, with a
// DC notch. z = (dB - col_mean) / col_std.
__global__ void local_threshold_kernel(const float* __restrict__ db,
                                       const float* __restrict__ col_mean,
                                       const float* __restrict__ col_std,
                                       uint8_t* __restrict__ high,
                                       uint8_t* __restrict__ low,
                                       int rows,
                                       int cols,
                                       float z_high,
                                       float z_low,
                                       int dc_center,
                                       int dc_notch) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const size_t idx = static_cast<size_t>(row) * cols + col;
  if (dc_notch > 0 && abs(col - dc_center) <= dc_notch) {
    high[idx] = 0;
    low[idx] = 0;
    return;
  }
  const float z = (db[idx] - col_mean[col]) / col_std[col];
  high[idx] = (z > z_high) ? 255 : 0;
  low[idx] = (z > z_low) ? 255 : 0;
}

// Line (1-D) erosion: min over [-radius, radius] along one axis.
//   axis 0 => along columns (horizontal line); axis 1 => along rows (vertical).
__global__ void erode_line_kernel(const uint8_t* __restrict__ in,
                                  uint8_t* __restrict__ out,
                                  int rows,
                                  int cols,
                                  int radius,
                                  int axis) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  uint8_t v = 255;
  for (int d = -radius; d <= radius; ++d) {
    const int r = (axis == 1) ? row + d : row;
    const int c = (axis == 0) ? col + d : col;
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      v = min(v, in[static_cast<size_t>(r) * cols + c]);
    }
  }
  out[static_cast<size_t>(row) * cols + col] = v;
}

// Line (1-D) dilation: max over [-radius, radius] along one axis.
__global__ void dilate_line_kernel(const uint8_t* __restrict__ in,
                                   uint8_t* __restrict__ out,
                                   int rows,
                                   int cols,
                                   int radius,
                                   int axis) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  uint8_t v = 0;
  for (int d = -radius; d <= radius; ++d) {
    const int r = (axis == 1) ? row + d : row;
    const int c = (axis == 0) ? col + d : col;
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
      v = max(v, in[static_cast<size_t>(r) * cols + c]);
    }
  }
  out[static_cast<size_t>(row) * cols + col] = v;
}

// Elementwise logical OR of two binary images.
__global__ void or_kernel(const uint8_t* __restrict__ a,
                          const uint8_t* __restrict__ b,
                          uint8_t* __restrict__ out,
                          size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = (a[i] || b[i]) ? 255 : 0;
}

// Block-reduced sum and sum-of-squares of a float image (edge gradient stats).
__global__ void image_stats_kernel(const float* __restrict__ data,
                                   size_t n,
                                   double* __restrict__ stats) {
  __shared__ double s_sum[kReduceThreads];
  __shared__ double s_sumsq[kReduceThreads];
  const unsigned tid = threadIdx.x;
  double local_sum = 0.0, local_sumsq = 0.0;
  for (size_t i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
    const double x = data[i];
    local_sum += x;
    local_sumsq += x * x;
  }
  s_sum[tid] = local_sum;
  s_sumsq[tid] = local_sumsq;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid] += s_sum[tid + stride];
      s_sumsq[tid] += s_sumsq[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(&stats[0], s_sum[0]);
    atomicAdd(&stats[1], s_sumsq[0]);
  }
}

__global__ void threshold_kernel(const float* __restrict__ data,
                                 uint8_t* __restrict__ out,
                                 size_t n,
                                 float cutoff) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = (data[i] > cutoff) ? 255 : 0;
}

__device__ __forceinline__ float clamped_at(const float* __restrict__ img,
                                             int r,
                                             int c,
                                             int rows,
                                             int cols) {
  const int rr = min(max(r, 0), rows - 1);
  const int cc = min(max(c, 0), cols - 1);
  return img[static_cast<size_t>(rr) * cols + cc];
}

__global__ void sobel_kernel(const float* __restrict__ img,
                             float* __restrict__ grad,
                             int rows,
                             int cols) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const float gx = -clamped_at(img, row - 1, col - 1, rows, cols) -
                   2.0f * clamped_at(img, row, col - 1, rows, cols) -
                   clamped_at(img, row + 1, col - 1, rows, cols) +
                   clamped_at(img, row - 1, col + 1, rows, cols) +
                   2.0f * clamped_at(img, row, col + 1, rows, cols) +
                   clamped_at(img, row + 1, col + 1, rows, cols);
  const float gy = -clamped_at(img, row - 1, col - 1, rows, cols) -
                   2.0f * clamped_at(img, row - 1, col, rows, cols) -
                   clamped_at(img, row - 1, col + 1, rows, cols) +
                   clamped_at(img, row + 1, col - 1, rows, cols) +
                   2.0f * clamped_at(img, row + 1, col, rows, cols) +
                   clamped_at(img, row + 1, col + 1, rows, cols);
  grad[static_cast<size_t>(row) * cols + col] = sqrtf(gx * gx + gy * gy);
}

// --- Connected-component labeling (label equivalence + path compression) ---

__global__ void ccl_init_kernel(const uint8_t* __restrict__ fg,
                               int32_t* __restrict__ labels,
                               size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  labels[i] = (fg[i] != 0) ? static_cast<int32_t>(i) : -1;
}

__global__ void ccl_propagate_kernel(int32_t* __restrict__ labels,
                                     int rows,
                                     int cols,
                                     int32_t* __restrict__ changed) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const size_t idx = static_cast<size_t>(row) * cols + col;
  int32_t cur = labels[idx];
  if (cur < 0) {
    return;
  }
  int32_t best = cur;
  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      if (dr == 0 && dc == 0) {
        continue;
      }
      const int r = row + dr;
      const int c = col + dc;
      if (r >= 0 && r < rows && c >= 0 && c < cols) {
        const int32_t nl = labels[static_cast<size_t>(r) * cols + c];
        if (nl >= 0 && nl < best) {
          best = nl;
        }
      }
    }
  }
  if (best < cur) {
    labels[idx] = best;
    *changed = 1;
  }
}

__global__ void ccl_compress_kernel(int32_t* __restrict__ labels, size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  int32_t l = labels[i];
  if (l < 0) {
    return;
  }
  int guard = 0;
  while (labels[l] != l && guard++ < 64) {
    l = labels[l];
  }
  labels[i] = l;
}

__global__ void ccl_area_kernel(const int32_t* __restrict__ labels,
                               int32_t* __restrict__ areas,
                               size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int32_t l = labels[i];
  if (l >= 0) {
    atomicAdd(&areas[l], 1);
  }
}

// Mark, per root label, whether the component contains any high-threshold seed.
__global__ void ccl_seed_kernel(const int32_t* __restrict__ labels,
                               const uint8_t* __restrict__ high,
                               int32_t* __restrict__ seeds,
                               size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int32_t l = labels[i];
  if (l >= 0 && high[i] != 0) {
    seeds[l] = 1;
  }
}

// Emit final mask. Blob = component that (a) contains a seed [hysteresis] and
// (b) meets the minimum area. combine_mode: 0 = blob, 1 = blob OR edge, 2 = edge.
__global__ void combine_kernel(const int32_t* __restrict__ labels,
                              const int32_t* __restrict__ areas,
                              const int32_t* __restrict__ seeds,
                              const uint8_t* __restrict__ edges,
                              uint8_t* __restrict__ out,
                              int rows,
                              int cols,
                              int min_area,
                              int mode,
                              int dc_center,
                              int dc_notch) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  const size_t i = static_cast<size_t>(row) * cols + col;
  if (dc_notch > 0 && abs(col - dc_center) <= dc_notch) {
    out[i] = 0;
    return;
  }
  const int32_t l = labels[i];
  const bool blob = (l >= 0) && (seeds[l] != 0) && (areas[l] >= min_area);
  const bool edge = edges[i] != 0;
  bool keep = false;
  switch (mode) {
    case 2: keep = edge; break;
    case 1: keep = blob || edge; break;
    default: keep = blob; break;
  }
  out[i] = keep ? 255 : 0;
}

}  // namespace

void ComputerVisionTuned::setup(OperatorSpec& spec) {
  spec.input<computer_vision_tuned_in_t>("in", holoscan::IOSpec::IOSize{16});
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(num_channels_, "num_channels", "Number of channels",
             "Pipeline channel count (routing validation).", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter",
             "Channel index this operator instance handles.", 0);
  spec.param(z_high_, "z_high", "Seed z-score",
             "Hysteresis seed threshold: sigma above the per-column local background.", 4.0f);
  spec.param(z_low_, "z_low", "Grow z-score",
             "Hysteresis grow threshold: sigma above the local background; low-threshold "
             "regions are kept only if connected to a seed.", 2.0f);
  spec.param(clip_z_, "clip_z", "Background sigma-clip",
             "Sigma-clip cut used when estimating the per-column background (rejects signal).",
             3.0f);
  spec.param(morph_radius_, "morph_radius", "Morphology radius",
             "Line structuring-element radius (bins/rows); 1 => length-3 lines.", 1);
  spec.param(close_iterations_, "close_iterations", "Closing iterations",
             "Separable (square) morphological closing passes to fill gaps.", 1);
  spec.param(edge_zscore_, "edge_zscore", "Edge z-score",
             "Sobel gradient-magnitude threshold in z-score units (only used when "
             "combine_mode includes edges).", 4.0f);
  spec.param(min_blob_area_, "min_blob_area", "Minimum blob area",
             "Connected components smaller than this many pixels are discarded.", 24);
  spec.param(ccl_max_iterations_, "ccl_max_iterations", "CCL max iterations",
             "Safety cap on label-propagation sweeps.", 256);
  spec.param(min_std_db_, "min_std_db", "Minimum std (dB)",
             "Floor on the per-column background std to avoid divide-by-noise.", 0.5f);
  spec.param(dc_notch_bins_, "dc_notch_bins", "DC notch bins",
             "Bins each side of band center forced to no-detect (DC / LO leakage).", 4);
  spec.param(combine_mode_, "combine_mode", "Combine mode",
             "Final mask composition: 'blob' (default), 'blob_or_edge', or 'edge'.",
             std::string("blob"));
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one mask every N frames.", 1);
}

void ComputerVisionTuned::initialize() {
  holoscan::Operator::initialize();
}

void ComputerVisionTuned::compute(InputContext& op_input,
                                  OutputContext& op_output,
                                  ExecutionContext&) {
  auto maybe_input = op_input.receive<computer_vision_tuned_in_t>("in");
  if (!maybe_input) {
    return;
  }
  auto input = maybe_input.value();
  auto& in_tensor = std::get<0>(input);
  cudaStream_t stream = std::get<1>(input);

  const int rows = static_cast<int>(in_tensor.Size(0));
  const int cols = static_cast<int>(in_tensor.Size(1));
  if (rows <= 0 || cols <= 0) {
    return;
  }
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);

  if (!state_.allocated) {
    throw_if_cuda_error(cudaMalloc(&state_.db_image, n * sizeof(float)), "cudaMalloc(db_image)");
    throw_if_cuda_error(cudaMalloc(&state_.scratch_f, n * sizeof(float)), "cudaMalloc(scratch_f)");
    throw_if_cuda_error(cudaMalloc(&state_.col_mean, cols * sizeof(float)), "cudaMalloc(col_mean)");
    throw_if_cuda_error(cudaMalloc(&state_.col_std, cols * sizeof(float)), "cudaMalloc(col_std)");
    throw_if_cuda_error(cudaMalloc(&state_.high, n * sizeof(uint8_t)), "cudaMalloc(high)");
    throw_if_cuda_error(cudaMalloc(&state_.low, n * sizeof(uint8_t)), "cudaMalloc(low)");
    throw_if_cuda_error(cudaMalloc(&state_.m1, n * sizeof(uint8_t)), "cudaMalloc(m1)");
    throw_if_cuda_error(cudaMalloc(&state_.m2, n * sizeof(uint8_t)), "cudaMalloc(m2)");
    throw_if_cuda_error(cudaMalloc(&state_.m3, n * sizeof(uint8_t)), "cudaMalloc(m3)");
    throw_if_cuda_error(cudaMalloc(&state_.edges, n * sizeof(uint8_t)), "cudaMalloc(edges)");
    throw_if_cuda_error(cudaMalloc(&state_.labels, n * sizeof(int32_t)), "cudaMalloc(labels)");
    throw_if_cuda_error(cudaMalloc(&state_.areas, n * sizeof(int32_t)), "cudaMalloc(areas)");
    throw_if_cuda_error(cudaMalloc(&state_.seeds, n * sizeof(int32_t)), "cudaMalloc(seeds)");
    throw_if_cuda_error(cudaMalloc(&state_.changed, sizeof(int32_t)), "cudaMalloc(changed)");
    throw_if_cuda_error(cudaMalloc(&state_.stats, 2 * sizeof(double)), "cudaMalloc(stats)");
    state_.allocated = true;
  }

  ++frame_number_;

  auto meta = metadata();
  const uint16_t channel_number =
      meta ? meta->get<uint16_t>("channel_number", static_cast<uint16_t>(channel_filter_.get()))
           : static_cast<uint16_t>(channel_filter_.get());

  const int stride = std::max(1, emit_stride_.get());
  const bool emit_this_frame = (frame_number_ % static_cast<uint64_t>(stride)) == 0;
  if (!emit_this_frame) {
    if (meta) {
      meta->set("computer_vision_tuned_emitted", false);
    }
    return;
  }

  const dim3 block2d(kBlockDim, kBlockDim);
  const dim3 grid2d((cols + block2d.x - 1) / block2d.x, (rows + block2d.y - 1) / block2d.y);
  const int block1d = 256;
  const int grid1d = static_cast<int>((n + block1d - 1) / block1d);
  const int grid1d_cols = (cols + block1d - 1) / block1d;
  const int dc_center = cols / 2;
  const int dc_notch = std::max(0, dc_notch_bins_.get());
  const int radius = std::max(1, morph_radius_.get());

  // 1) dB image.
  complex_to_db_shift_kernel<<<grid2d, block2d, 0, stream>>>(in_tensor.Data(), state_.db_image,
                                                            rows, cols);
  throw_if_cuda_error(cudaGetLastError(), "complex_to_db_shift_kernel launch");

  // 2) Local per-column background + hysteresis dual threshold.
  column_background_kernel<<<grid1d_cols, block1d, 0, stream>>>(
      state_.db_image, state_.col_mean, state_.col_std, rows, cols, clip_z_.get(), min_std_db_.get());
  throw_if_cuda_error(cudaGetLastError(), "column_background_kernel launch");
  local_threshold_kernel<<<grid2d, block2d, 0, stream>>>(
      state_.db_image, state_.col_mean, state_.col_std, state_.high, state_.low, rows, cols,
      z_high_.get(), z_low_.get(), dc_center, dc_notch);
  throw_if_cuda_error(cudaGetLastError(), "local_threshold_kernel launch");

  // 3) Direction-aware opening of the low mask: open_h OR open_v so thin
  //    horizontal (wideband burst) and thin vertical (narrowband carrier)
  //    structures both survive, while isolated speckle is removed.
  // open_h: erode then dilate along columns (axis 0). low -> m1 -> m2.
  erode_line_kernel<<<grid2d, block2d, 0, stream>>>(state_.low, state_.m1, rows, cols, radius, 0);
  throw_if_cuda_error(cudaGetLastError(), "erode_line(h) launch");
  dilate_line_kernel<<<grid2d, block2d, 0, stream>>>(state_.m1, state_.m2, rows, cols, radius, 0);
  throw_if_cuda_error(cudaGetLastError(), "dilate_line(h) launch");
  // open_v: erode then dilate along rows (axis 1). low -> m1 -> m3.
  erode_line_kernel<<<grid2d, block2d, 0, stream>>>(state_.low, state_.m1, rows, cols, radius, 1);
  throw_if_cuda_error(cudaGetLastError(), "erode_line(v) launch");
  dilate_line_kernel<<<grid2d, block2d, 0, stream>>>(state_.m1, state_.m3, rows, cols, radius, 1);
  throw_if_cuda_error(cudaGetLastError(), "dilate_line(v) launch");
  // cleaned = open_h OR open_v -> m1.
  or_kernel<<<grid1d, block1d, 0, stream>>>(state_.m2, state_.m3, state_.m1, n);
  throw_if_cuda_error(cudaGetLastError(), "or_kernel launch");

  // 4) Separable (square) closing to fill small gaps: dilate_h,dilate_v then
  //    erode_h,erode_v. Ping-pong m1 <-> m2.
  uint8_t* cur = state_.m1;
  uint8_t* other = state_.m2;
  for (int it = 0; it < std::max(0, close_iterations_.get()); ++it) {
    dilate_line_kernel<<<grid2d, block2d, 0, stream>>>(cur, other, rows, cols, radius, 0);
    throw_if_cuda_error(cudaGetLastError(), "close dilate(h) launch");
    std::swap(cur, other);
    dilate_line_kernel<<<grid2d, block2d, 0, stream>>>(cur, other, rows, cols, radius, 1);
    throw_if_cuda_error(cudaGetLastError(), "close dilate(v) launch");
    std::swap(cur, other);
    erode_line_kernel<<<grid2d, block2d, 0, stream>>>(cur, other, rows, cols, radius, 0);
    throw_if_cuda_error(cudaGetLastError(), "close erode(h) launch");
    std::swap(cur, other);
    erode_line_kernel<<<grid2d, block2d, 0, stream>>>(cur, other, rows, cols, radius, 1);
    throw_if_cuda_error(cudaGetLastError(), "close erode(v) launch");
    std::swap(cur, other);
  }
  // Re-inject the raw high seeds so a seed removed by the opening still anchors
  // its region (correct hysteresis connectivity = opened-low UNION strong-seeds;
  // isolated high noise spikes fall out later via the min-area filter).
  or_kernel<<<grid1d, block1d, 0, stream>>>(cur, state_.high, other, n);
  throw_if_cuda_error(cudaGetLastError(), "seed reinjection or_kernel launch");
  uint8_t* cleaned = other;  // final cleaned foreground (opened low ∪ high seeds)

  // 5) Optional Sobel edges (only needed if combine_mode uses them).
  const std::string cm = combine_mode_.get();
  int mode = 0;  // blob
  if (cm == "blob_or_edge") {
    mode = 1;
  } else if (cm == "edge") {
    mode = 2;
  }
  if (mode != 0) {
    sobel_kernel<<<grid2d, block2d, 0, stream>>>(state_.db_image, state_.scratch_f, rows, cols);
    throw_if_cuda_error(cudaGetLastError(), "sobel_kernel launch");
    throw_if_cuda_error(cudaMemsetAsync(state_.stats, 0, 2 * sizeof(double), stream),
                        "cudaMemsetAsync(stats)");
    const int reduce_blocks =
        std::min(1024, static_cast<int>((n + kReduceThreads - 1) / kReduceThreads));
    image_stats_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(state_.scratch_f, n,
                                                                    state_.stats);
    throw_if_cuda_error(cudaGetLastError(), "image_stats_kernel launch");
    double host_stats[2] = {0.0, 0.0};
    throw_if_cuda_error(cudaMemcpyAsync(host_stats, state_.stats, 2 * sizeof(double),
                                        cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync(stats)");
    throw_if_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stats)");
    const double dn = static_cast<double>(n);
    const double gm = host_stats[0] / dn;
    const double gv = host_stats[1] / dn - gm * gm;
    const float grad_std = static_cast<float>(sqrt(gv > 0.0 ? gv : 0.0));
    const float edge_cutoff = static_cast<float>(gm) + edge_zscore_.get() * std::max(grad_std, 1e-3f);
    threshold_kernel<<<grid1d, block1d, 0, stream>>>(state_.scratch_f, state_.edges, n, edge_cutoff);
    throw_if_cuda_error(cudaGetLastError(), "edge threshold_kernel launch");
  } else {
    throw_if_cuda_error(cudaMemsetAsync(state_.edges, 0, n, stream), "cudaMemset(edges)");
  }

  // 6) Connected components on the cleaned foreground; hysteresis seed + area.
  ccl_init_kernel<<<grid1d, block1d, 0, stream>>>(cleaned, state_.labels, n);
  throw_if_cuda_error(cudaGetLastError(), "ccl_init_kernel launch");
  const int max_iters = std::max(1, ccl_max_iterations_.get());
  for (int it = 0; it < max_iters; ++it) {
    throw_if_cuda_error(cudaMemsetAsync(state_.changed, 0, sizeof(int32_t), stream),
                        "cudaMemsetAsync(changed)");
    ccl_propagate_kernel<<<grid2d, block2d, 0, stream>>>(state_.labels, rows, cols, state_.changed);
    throw_if_cuda_error(cudaGetLastError(), "ccl_propagate_kernel launch");
    ccl_compress_kernel<<<grid1d, block1d, 0, stream>>>(state_.labels, n);
    throw_if_cuda_error(cudaGetLastError(), "ccl_compress_kernel launch");
    int32_t host_changed = 0;
    throw_if_cuda_error(cudaMemcpyAsync(&host_changed, state_.changed, sizeof(int32_t),
                                        cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync(changed)");
    throw_if_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize(changed)");
    if (host_changed == 0) {
      break;
    }
  }
  throw_if_cuda_error(cudaMemsetAsync(state_.areas, 0, n * sizeof(int32_t), stream),
                      "cudaMemsetAsync(areas)");
  throw_if_cuda_error(cudaMemsetAsync(state_.seeds, 0, n * sizeof(int32_t), stream),
                      "cudaMemsetAsync(seeds)");
  ccl_area_kernel<<<grid1d, block1d, 0, stream>>>(state_.labels, state_.areas, n);
  throw_if_cuda_error(cudaGetLastError(), "ccl_area_kernel launch");
  ccl_seed_kernel<<<grid1d, block1d, 0, stream>>>(state_.labels, state_.high, state_.seeds, n);
  throw_if_cuda_error(cudaGetLastError(), "ccl_seed_kernel launch");

  // 7) Final mask into an owned device buffer handed downstream.
  auto mask_device = make_owned_device_u8(n);
  combine_kernel<<<grid2d, block2d, 0, stream>>>(state_.labels, state_.areas, state_.seeds,
                                                state_.edges, mask_device.get(), rows, cols,
                                                std::max(1, min_blob_area_.get()), mode, dc_center,
                                                dc_notch);
  throw_if_cuda_error(cudaGetLastError(), "combine_kernel launch");
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
    meta->set("computer_vision_tuned_emitted", true);
    meta->set("computer_vision_tuned_combine_mode", cm);
    meta->set("computer_vision_tuned_frame_number", frame_number_);
  }
}

void ComputerVisionTuned::free_device_state() {
  auto free_ptr = [](auto*& p) {
    if (p != nullptr) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_ptr(state_.db_image);
  free_ptr(state_.scratch_f);
  free_ptr(state_.col_mean);
  free_ptr(state_.col_std);
  free_ptr(state_.high);
  free_ptr(state_.low);
  free_ptr(state_.m1);
  free_ptr(state_.m2);
  free_ptr(state_.m3);
  free_ptr(state_.edges);
  free_ptr(state_.labels);
  free_ptr(state_.areas);
  free_ptr(state_.seeds);
  free_ptr(state_.changed);
  free_ptr(state_.stats);
  state_.allocated = false;
}

void ComputerVisionTuned::stop() {
  HOLOSCAN_LOG_INFO("computer_vision_tuned ch={} processed_frames={} emitted_masks={} mode={}",
                    channel_filter_.get(), frame_number_, detections_emitted_, combine_mode_.get());
  free_device_state();
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

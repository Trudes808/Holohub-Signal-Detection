// SPDX-FileCopyrightText: 2026 The University of Texas at Austin
//
// SPDX-License-Identifier: Apache-2.0
#include "computer_vision_baseline.hpp"

// Shared detector output contract (see cuda_dino_detector for the same include).
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
constexpr int kBlockDim = 16;      // 2D kernels
constexpr int kReduceThreads = 256;

void throw_if_cuda_error(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("computer_vision_baseline CUDA error during ") + what +
                             ": " + cudaGetErrorString(status));
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
  const complex v = in[row * cols + src_col];
  const float re = v.real();
  const float im = v.imag();
  const float power = re * re + im * im + kPowerEps;
  out_db[row * cols + col] = 10.0f * log10f(power);
}

// Block-reduced sum and sum-of-squares of a float image into stats[0], stats[1].
__global__ void image_stats_kernel(const float* __restrict__ data,
                                    size_t n,
                                    double* __restrict__ stats) {
  __shared__ double s_sum[kReduceThreads];
  __shared__ double s_sumsq[kReduceThreads];

  const unsigned tid = threadIdx.x;
  double local_sum = 0.0;
  double local_sumsq = 0.0;
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

// Adaptive threshold: foreground where value > mean + z*std.
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

// Morphological erosion: a pixel is foreground only if all in-bounds neighbors
// within the structuring element are foreground (min over the window).
__global__ void erode_kernel(const uint8_t* __restrict__ in,
                             uint8_t* __restrict__ out,
                             int rows,
                             int cols,
                             int radius) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  uint8_t v = 255;
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      const int r = row + dr;
      const int c = col + dc;
      if (r >= 0 && r < rows && c >= 0 && c < cols) {
        v = min(v, in[static_cast<size_t>(r) * cols + c]);
      }
    }
  }
  out[static_cast<size_t>(row) * cols + col] = v;
}

// Morphological dilation: max over the structuring element.
__global__ void dilate_kernel(const uint8_t* __restrict__ in,
                              uint8_t* __restrict__ out,
                              int rows,
                              int cols,
                              int radius) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= rows || col >= cols) {
    return;
  }
  uint8_t v = 0;
  for (int dr = -radius; dr <= radius; ++dr) {
    for (int dc = -radius; dc <= radius; ++dc) {
      const int r = row + dr;
      const int c = col + dc;
      if (r >= 0 && r < rows && c >= 0 && c < cols) {
        v = max(v, in[static_cast<size_t>(r) * cols + c]);
      }
    }
  }
  out[static_cast<size_t>(row) * cols + col] = v;
}

// Clamped (replicate-border) fetch from a float image.
__device__ __forceinline__ float clamped_at(const float* __restrict__ img,
                                             int r,
                                             int c,
                                             int rows,
                                             int cols) {
  const int rr = min(max(r, 0), rows - 1);
  const int cc = min(max(c, 0), cols - 1);
  return img[static_cast<size_t>(rr) * cols + cc];
}

// Sobel gradient magnitude on the dB image (edge strength).
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

// One propagation sweep: each foreground pixel adopts the minimum label among
// itself and its 8-connected foreground neighbors. Sets *changed if it moved.
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

// Path compression: chase the label chain toward its root.
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

// Emit final detection mask combining blobs (area-filtered) and/or edges.
//   mode 0 = blob only, mode 1 = blob OR edge, mode 2 = edge only.
__global__ void combine_kernel(const int32_t* __restrict__ labels,
                               const int32_t* __restrict__ areas,
                               const uint8_t* __restrict__ edges,
                               uint8_t* __restrict__ out,
                               size_t n,
                               int min_area,
                               int mode) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const int32_t l = labels[i];
  const bool blob = (l >= 0) && (areas[l] >= min_area);
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

void ComputerVisionBaseline::setup(OperatorSpec& spec) {
  spec.input<computer_vision_in_t>("in", holoscan::IOSpec::IOSize{16});
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(num_channels_, "num_channels", "Number of channels",
             "Number of channels in the pipeline (routing validation).", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter",
             "Channel index this operator instance handles.", 0);
  spec.param(threshold_zscore_, "threshold_zscore", "Threshold z-score",
             "Foreground when a pixel's dB exceeds the image mean by this many standard "
             "deviations. System-agnostic; not an absolute dB threshold.",
             3.0f);
  spec.param(morph_radius_, "morph_radius", "Morphology radius",
             "Structuring-element radius in bins (1 => 3x3).", 1);
  spec.param(open_iterations_, "open_iterations", "Opening iterations",
             "Morphological opening passes (erode then dilate) to remove speckle.", 1);
  spec.param(close_iterations_, "close_iterations", "Closing iterations",
             "Morphological closing passes (dilate then erode) to fill gaps.", 1);
  spec.param(edge_zscore_, "edge_zscore", "Edge z-score",
             "Sobel gradient-magnitude threshold in z-score units.", 3.0f);
  spec.param(min_blob_area_, "min_blob_area", "Minimum blob area",
             "Connected components smaller than this many pixels are discarded.", 32);
  spec.param(ccl_max_iterations_, "ccl_max_iterations", "CCL max iterations",
             "Safety cap on connected-component label-propagation sweeps.", 256);
  spec.param(combine_mode_, "combine_mode", "Combine mode",
             "Final mask composition: 'blob', 'blob_or_edge', or 'edge'.",
             std::string("blob_or_edge"));
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one mask every N frames.", 1);
}

void ComputerVisionBaseline::initialize() {
  holoscan::Operator::initialize();
}

void ComputerVisionBaseline::compute(InputContext& op_input,
                                     OutputContext& op_output,
                                     ExecutionContext&) {
  auto maybe_input = op_input.receive<computer_vision_in_t>("in");
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

  // Lazily allocate device buffers sized to the spectrogram.
  if (!state_.allocated) {
    throw_if_cuda_error(cudaMalloc(&state_.db_image, n * sizeof(float)), "cudaMalloc(db_image)");
    throw_if_cuda_error(cudaMalloc(&state_.scratch_f, n * sizeof(float)), "cudaMalloc(scratch_f)");
    throw_if_cuda_error(cudaMalloc(&state_.binary, n * sizeof(uint8_t)), "cudaMalloc(binary)");
    throw_if_cuda_error(cudaMalloc(&state_.morph_a, n * sizeof(uint8_t)), "cudaMalloc(morph_a)");
    throw_if_cuda_error(cudaMalloc(&state_.morph_b, n * sizeof(uint8_t)), "cudaMalloc(morph_b)");
    throw_if_cuda_error(cudaMalloc(&state_.edges, n * sizeof(uint8_t)), "cudaMalloc(edges)");
    throw_if_cuda_error(cudaMalloc(&state_.labels, n * sizeof(int32_t)), "cudaMalloc(labels)");
    throw_if_cuda_error(cudaMalloc(&state_.areas, n * sizeof(int32_t)), "cudaMalloc(areas)");
    throw_if_cuda_error(cudaMalloc(&state_.stats, 2 * sizeof(double)), "cudaMalloc(stats)");
    throw_if_cuda_error(cudaMalloc(&state_.changed, sizeof(int32_t)), "cudaMalloc(changed)");
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
      meta->set("computer_vision_emitted", false);
    }
    return;
  }

  const dim3 block2d(kBlockDim, kBlockDim);
  const dim3 grid2d((cols + block2d.x - 1) / block2d.x, (rows + block2d.y - 1) / block2d.y);
  const int block1d = 256;
  const int grid1d = static_cast<int>((n + block1d - 1) / block1d);

  // Helper: compute global mean/std of a float buffer on this stream.
  auto mean_std = [&](const float* data, float& mean, float& std) {
    throw_if_cuda_error(cudaMemsetAsync(state_.stats, 0, 2 * sizeof(double), stream),
                        "cudaMemsetAsync(stats)");
    const int reduce_blocks = std::min(1024, static_cast<int>((n + kReduceThreads - 1) / kReduceThreads));
    image_stats_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(data, n, state_.stats);
    throw_if_cuda_error(cudaGetLastError(), "image_stats_kernel launch");
    double host_stats[2] = {0.0, 0.0};
    throw_if_cuda_error(
        cudaMemcpyAsync(host_stats, state_.stats, 2 * sizeof(double), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(stats)");
    throw_if_cuda_error(cudaStreamSynchronize(stream), "cudaStreamSynchronize(stats)");
    const double dn = static_cast<double>(n);
    const double m = host_stats[0] / dn;
    const double var = host_stats[1] / dn - m * m;
    mean = static_cast<float>(m);
    std = static_cast<float>(sqrt(var > 0.0 ? var : 0.0));
  };

  // 1) dB image.
  complex_to_db_shift_kernel<<<grid2d, block2d, 0, stream>>>(in_tensor.Data(), state_.db_image, rows, cols);
  throw_if_cuda_error(cudaGetLastError(), "complex_to_db_shift_kernel launch");

  // 2) Adaptive z-score threshold -> binary foreground.
  float img_mean = 0.0f, img_std = 0.0f;
  mean_std(state_.db_image, img_mean, img_std);
  const float fg_cutoff = img_mean + threshold_zscore_.get() * std::max(img_std, 1e-3f);
  threshold_kernel<<<grid1d, block1d, 0, stream>>>(state_.db_image, state_.binary, n, fg_cutoff);
  throw_if_cuda_error(cudaGetLastError(), "threshold_kernel launch");

  // 3) Morphological opening (erode->dilate) then closing (dilate->erode).
  // Each primitive is one ping-pong step that reads `src` and writes a distinct
  // `dst`; `binary` is only ever read, and src/dst never alias. `src` holds the
  // cleaned foreground when the loops finish.
  const int radius = std::max(1, morph_radius_.get());
  uint8_t* src = state_.binary;
  uint8_t* dst = state_.morph_a;
  auto morph_step = [&](bool is_erode) {
    if (is_erode) {
      erode_kernel<<<grid2d, block2d, 0, stream>>>(src, dst, rows, cols, radius);
      throw_if_cuda_error(cudaGetLastError(), "erode_kernel launch");
    } else {
      dilate_kernel<<<grid2d, block2d, 0, stream>>>(src, dst, rows, cols, radius);
      throw_if_cuda_error(cudaGetLastError(), "dilate_kernel launch");
    }
    // Advance: the just-written buffer becomes the source; the next destination
    // is the other scratch buffer (never `binary`, never the current source).
    uint8_t* next_free = (dst == state_.morph_a) ? state_.morph_b : state_.morph_a;
    src = dst;
    dst = next_free;
  };

  for (int it = 0; it < std::max(0, open_iterations_.get()); ++it) {
    morph_step(true);   // erode
    morph_step(false);  // dilate
  }
  for (int it = 0; it < std::max(0, close_iterations_.get()); ++it) {
    morph_step(false);  // dilate
    morph_step(true);   // erode
  }
  uint8_t* cleaned = src;  // cleaned foreground mask

  // 4) Sobel edges on the dB image, thresholded by their own z-score.
  sobel_kernel<<<grid2d, block2d, 0, stream>>>(state_.db_image, state_.scratch_f, rows, cols);
  throw_if_cuda_error(cudaGetLastError(), "sobel_kernel launch");
  float grad_mean = 0.0f, grad_std = 0.0f;
  mean_std(state_.scratch_f, grad_mean, grad_std);
  const float edge_cutoff = grad_mean + edge_zscore_.get() * std::max(grad_std, 1e-3f);
  threshold_kernel<<<grid1d, block1d, 0, stream>>>(state_.scratch_f, state_.edges, n, edge_cutoff);
  throw_if_cuda_error(cudaGetLastError(), "edge threshold_kernel launch");

  // 5) Connected components on the cleaned foreground, then area filter.
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
  ccl_area_kernel<<<grid1d, block1d, 0, stream>>>(state_.labels, state_.areas, n);
  throw_if_cuda_error(cudaGetLastError(), "ccl_area_kernel launch");

  // 6) Final mask into an owned device buffer handed downstream.
  auto mask_device = make_owned_device_u8(n);
  int mode = 1;  // blob_or_edge
  const std::string cm = combine_mode_.get();
  if (cm == "blob") {
    mode = 0;
  } else if (cm == "edge") {
    mode = 2;
  }
  combine_kernel<<<grid1d, block1d, 0, stream>>>(state_.labels,
                                                 state_.areas,
                                                 state_.edges,
                                                 mask_device.get(),
                                                 n,
                                                 std::max(1, min_blob_area_.get()),
                                                 mode);
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
    meta->set("computer_vision_emitted", true);
    meta->set("computer_vision_combine_mode", cm);
    meta->set("computer_vision_frame_number", frame_number_);
  }
}

void ComputerVisionBaseline::free_device_state() {
  auto free_ptr = [](auto*& p) {
    if (p != nullptr) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_ptr(state_.db_image);
  free_ptr(state_.scratch_f);
  free_ptr(state_.binary);
  free_ptr(state_.morph_a);
  free_ptr(state_.morph_b);
  free_ptr(state_.edges);
  free_ptr(state_.labels);
  free_ptr(state_.areas);
  free_ptr(state_.stats);
  free_ptr(state_.changed);
  state_.allocated = false;
}

void ComputerVisionBaseline::stop() {
  HOLOSCAN_LOG_INFO("computer_vision_baseline ch={} processed_frames={} emitted_masks={} mode={}",
                    channel_filter_.get(), frame_number_, detections_emitted_, combine_mode_.get());
  free_device_state();
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

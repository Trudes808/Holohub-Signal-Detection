// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "signal_snip_core.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace holoscan::ops {

// ---------------------------------------------------------------------------------------------
// DeviceBufferPool
// ---------------------------------------------------------------------------------------------

uint64_t DeviceBufferPool::bucket_for(uint64_t elements) {
  // Round up to a coarse granularity (not a power of two): full IQ frames are all the same size, so
  // they reuse exactly, while the pow2 rounding would nearly double the ~42 MB/frame footprint.
  constexpr uint64_t kGranularity = 1ULL << 16;  // 65536 elements
  const uint64_t rounded = ((std::max<uint64_t>(1, elements) + kGranularity - 1) / kGranularity) * kGranularity;
  return rounded;
}

std::shared_ptr<SnipComplex> DeviceBufferPool::acquire(uint64_t elements) {
  const uint64_t capacity = bucket_for(std::max<uint64_t>(1, elements));

  SnipComplex* raw = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& bucket : buckets_) {
      if (bucket.capacity_elements == capacity && !bucket.free_list.empty()) {
        raw = bucket.free_list.back();
        bucket.free_list.pop_back();
        break;
      }
    }
  }

  if (raw == nullptr) {
    if (cudaMalloc(&raw, capacity * sizeof(SnipComplex)) != cudaSuccess || raw == nullptr) {
      throw std::runtime_error("DeviceBufferPool: cudaMalloc failed");
    }
  }

  // The deleter returns the buffer to the free list and keeps the pool alive until the last snippet
  // that references any of its buffers is destroyed (the file sink drops the final reference).
  auto self = shared_from_this();
  return std::shared_ptr<SnipComplex>(raw, [self, capacity](SnipComplex* ptr) {
    if (ptr == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(self->mutex_);
    for (auto& bucket : self->buckets_) {
      if (bucket.capacity_elements == capacity) {
        bucket.free_list.push_back(ptr);
        return;
      }
    }
    self->buckets_.push_back(DeviceBufferPool::Bucket{capacity, {ptr}});
  });
}

DeviceBufferPool::~DeviceBufferPool() {
  for (auto& bucket : buckets_) {
    for (SnipComplex* ptr : bucket.free_list) {
      cudaFree(ptr);
    }
  }
}

namespace snip {

// ---------------------------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------------------------

std::vector<BoundingBox> label_components(const std::vector<uint8_t>& mask,
                                          int rows,
                                          int cols,
                                          int min_pixels,
                                          CcScratch& scratch) {
  std::vector<BoundingBox> boxes;
  if (mask.empty() || rows <= 0 || cols <= 0) {
    return boxes;
  }

  const size_t n = mask.size();
  // Reused, allocation-free scratch: visited flags + a flat-index FIFO (head/tail into `queue`).
  scratch.visited.assign(n, 0);
  if (scratch.queue.size() < n) {
    scratch.queue.resize(n);
  }
  uint8_t* visited = scratch.visited.data();
  int* queue = scratch.queue.data();

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int seed = row * cols + col;
      if (!mask[seed] || visited[seed]) {
        continue;
      }
      BoundingBox box;
      box.row0 = box.row1 = row;
      box.col0 = box.col1 = col;
      box.pixel_count = 0;

      int head = 0;
      int tail = 0;
      queue[tail++] = seed;
      visited[seed] = 1;
      while (head < tail) {
        const int idx = queue[head++];
        const int cr = idx / cols;
        const int cc = idx - cr * cols;
        ++box.pixel_count;
        box.row0 = std::min(box.row0, cr);
        box.row1 = std::max(box.row1, cr);
        box.col0 = std::min(box.col0, cc);
        box.col1 = std::max(box.col1, cc);
        // 4-connected neighbors (inlined to avoid per-pixel array iteration).
        if (cr + 1 < rows) {
          const int ni = idx + cols;
          if (mask[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
        }
        if (cr - 1 >= 0) {
          const int ni = idx - cols;
          if (mask[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
        }
        if (cc + 1 < cols) {
          const int ni = idx + 1;
          if (mask[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
        }
        if (cc - 1 >= 0) {
          const int ni = idx - 1;
          if (mask[ni] && !visited[ni]) { visited[ni] = 1; queue[tail++] = ni; }
        }
      }

      if (box.pixel_count >= min_pixels) {
        boxes.push_back(box);
      }
    }
  }
  return boxes;
}

std::vector<BoundingBox> merge_boxes(std::vector<BoundingBox> boxes, int gap_rows, int gap_cols) {
  if (gap_rows < 0) gap_rows = 0;
  if (gap_cols < 0) gap_cols = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    std::vector<BoundingBox> merged;
    for (const auto& b : boxes) {
      bool absorbed = false;
      for (auto& r : merged) {
        // Overlap test with each box expanded by the gap tolerance.
        const bool row_touch = b.row0 <= r.row1 + gap_rows && r.row0 <= b.row1 + gap_rows;
        const bool col_touch = b.col0 <= r.col1 + gap_cols && r.col0 <= b.col1 + gap_cols;
        if (row_touch && col_touch) {
          r.row0 = std::min(r.row0, b.row0);
          r.row1 = std::max(r.row1, b.row1);
          r.col0 = std::min(r.col0, b.col0);
          r.col1 = std::max(r.col1, b.col1);
          r.pixel_count += b.pixel_count;
          absorbed = true;
          changed = true;
          break;
        }
      }
      if (!absorbed) {
        merged.push_back(b);
      }
    }
    boxes = std::move(merged);
  }
  return boxes;
}

PhysicalRegion map_box_to_physical(const BoundingBox& box, const FrameGeometry& geom) {
  PhysicalRegion region;
  const double rows = std::max(1, geom.mask_rows);
  const double cols = std::max(1, geom.mask_cols);

  // Time: mask rows -> full-rate sample range within the frame.
  const double frame_samples = static_cast<double>(geom.frame_sample_count);
  uint64_t local_start = static_cast<uint64_t>(std::floor((box.row0 / rows) * frame_samples));
  uint64_t local_end = static_cast<uint64_t>(std::ceil(((box.row1 + 1) / rows) * frame_samples));
  local_end = std::min<uint64_t>(local_end, geom.frame_sample_count);
  local_start = std::min<uint64_t>(local_start, local_end);

  region.local_start = local_start;
  region.sample_start = geom.frame_sample_start + local_start;
  region.sample_count = local_end - local_start;

  // Frequency: mask cols -> absolute RF Hz across the band centered on center_freq_hz.
  const double fs = geom.sample_rate_hz;
  region.freq_lower_hz = geom.center_freq_hz + ((box.col0 / cols) - 0.5) * fs;
  region.freq_upper_hz = geom.center_freq_hz + (((box.col1 + 1) / cols) - 0.5) * fs;
  region.freq_center_hz = 0.5 * (region.freq_lower_hz + region.freq_upper_hz);
  region.bandwidth_hz = region.freq_upper_hz - region.freq_lower_hz;
  region.sample_rate_hz = fs;
  region.center_freq_hz = geom.center_freq_hz;
  return region;
}

// ---------------------------------------------------------------------------------------------
// DDC (mix -> low-pass FIR -> decimate), fused in one kernel
// ---------------------------------------------------------------------------------------------

namespace {

// Windowed-sinc low-pass prototype (Hamming window). cutoff_norm in cycles/sample (0..0.5).
std::vector<float> design_lowpass(int num_taps, double cutoff_norm) {
  if (num_taps < 1) {
    num_taps = 1;
  }
  if ((num_taps % 2) == 0) {
    ++num_taps;  // force odd for a symmetric type-I FIR
  }
  cutoff_norm = std::min(std::max(cutoff_norm, 1e-4), 0.4999);
  std::vector<float> taps(static_cast<size_t>(num_taps));
  const int mid = num_taps / 2;
  const double two_pi = 2.0 * M_PI;
  double sum = 0.0;
  for (int n = 0; n < num_taps; ++n) {
    const int k = n - mid;
    double sinc;
    if (k == 0) {
      sinc = 2.0 * cutoff_norm;
    } else {
      sinc = std::sin(two_pi * cutoff_norm * k) / (M_PI * k);
    }
    const double window = 0.54 - 0.46 * std::cos(two_pi * n / (num_taps - 1));
    const double tap = sinc * window;
    taps[static_cast<size_t>(n)] = static_cast<float>(tap);
    sum += tap;
  }
  if (sum != 0.0) {
    for (float& tap : taps) {
      tap = static_cast<float>(tap / sum);  // unity DC gain
    }
  }
  return taps;
}

// Mix to baseband: one sincos per INPUT sample (parallelized), written to a scratch buffer. This is
// the key efficiency point -- the old fused kernel recomputed the phase inside the FIR loop, i.e.
// num_taps sincos per OUTPUT sample (n_out * num_taps total). Mixing once first makes it n_in.
__global__ void mix_kernel(const SnipComplex* __restrict__ x,
                           int n_in,
                           float omega,  // = -2*pi*f_offset/fs
                           SnipComplex* __restrict__ mixed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_in) {
    return;
  }
  float sn;
  float cs;
  sincosf(omega * static_cast<float>(i), &sn, &cs);
  const SnipComplex s = x[i];
  mixed[i] = SnipComplex(s.real() * cs - s.imag() * sn, s.real() * sn + s.imag() * cs);
}

// Decimating low-pass FIR over the already-mixed signal: MACs only, no transcendentals. Taps are
// symmetric type-I; `mid` centers the window. Output m samples the mixed stream at m*decim.
__global__ void fir_decim_kernel(const SnipComplex* __restrict__ mixed,
                                 int n_in,
                                 const float* __restrict__ taps,
                                 int num_taps,
                                 int decim,
                                 SnipComplex* __restrict__ y,
                                 int n_out) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= n_out) {
    return;
  }
  const int mid = num_taps / 2;
  const int center = m * decim;
  float acc_re = 0.0f;
  float acc_im = 0.0f;
  for (int k = 0; k < num_taps; ++k) {
    const int idx = center + (mid - k);
    if (idx < 0 || idx >= n_in) {
      continue;
    }
    const SnipComplex s = mixed[idx];
    const float h = taps[k];
    acc_re += h * s.real();
    acc_im += h * s.imag();
  }
  y[m] = SnipComplex(acc_re, acc_im);
}

}  // namespace

SnippetIq ddc_extract(const SnipComplex* frame_iq,
                      uint64_t frame_n,
                      const PhysicalRegion& region,
                      const SnipDspParams& params,
                      DeviceBufferPool& pool,
                      cudaStream_t stream) {
  SnippetIq out;
  if (region.sample_count == 0 || region.local_start >= frame_n) {
    return out;
  }
  const uint64_t n_in = std::min<uint64_t>(region.sample_count, frame_n - region.local_start);
  if (n_in == 0) {
    return out;
  }

  const double fs = region.sample_rate_hz;
  const double detected_bw = std::max(region.bandwidth_hz, 1.0);
  const double keep_bw = (detected_bw + params.bandwidth_margin_hz) * (1.0 + params.oversample_percent / 100.0);

  int decim = 1;
  if (params.enable_downsample && fs > 0.0 && keep_bw > 0.0) {
    decim = static_cast<int>(std::floor(fs / keep_bw));
    decim = std::max(1, decim);
  }
  const double out_rate = fs / static_cast<double>(decim);

  // Low-pass cutoff (one-sided) as a fraction of the full sample rate.
  const double cutoff_norm = (fs > 0.0) ? (0.5 * keep_bw / fs) : 0.25;
  const std::vector<float> taps = design_lowpass(params.fir_num_taps, cutoff_norm);
  const int num_taps = static_cast<int>(taps.size());

  const uint64_t n_out = (n_in + static_cast<uint64_t>(decim) - 1) / static_cast<uint64_t>(decim);
  if (n_out == 0) {
    return out;
  }

  // Baseband mixing frequency: shift the signal's center to DC.
  const double f_offset = region.freq_center_hz - region.center_freq_hz;
  const float omega = (fs > 0.0) ? static_cast<float>(-2.0 * M_PI * f_offset / fs) : 0.0f;

  // Upload taps to the device (small, stream-ordered).
  float* taps_device = nullptr;
  if (cudaMallocAsync(&taps_device, num_taps * sizeof(float), stream) != cudaSuccess) {
    throw std::runtime_error("ddc_extract: cudaMallocAsync taps failed");
  }
  cudaMemcpyAsync(taps_device, taps.data(), num_taps * sizeof(float), cudaMemcpyHostToDevice, stream);

  const int threads = 256;
  // Mix once (n_in sincos), then decimating FIR (MACs only).
  auto mixed = pool.acquire(n_in);
  const int mix_blocks = static_cast<int>((n_in + threads - 1) / threads);
  mix_kernel<<<mix_blocks, threads, 0, stream>>>(frame_iq + region.local_start,
                                                 static_cast<int>(n_in), omega, mixed.get());

  auto device_out = pool.acquire(n_out);
  const int fir_blocks = static_cast<int>((n_out + threads - 1) / threads);
  fir_decim_kernel<<<fir_blocks, threads, 0, stream>>>(mixed.get(),
                                                       static_cast<int>(n_in),
                                                       taps_device,
                                                       num_taps,
                                                       decim,
                                                       device_out.get(),
                                                       static_cast<int>(n_out));
  cudaFreeAsync(taps_device, stream);
  // `mixed` returns to the pool when this shared_ptr drops; the FIR above has been enqueued on the
  // same stream, so its reads are ordered before any later reuse of the buffer on that stream.

  out.device_iq = std::move(device_out);
  out.n_iq = n_out;
  out.sample_rate_hz = out_rate;
  return out;
}

SnippetIq copy_time_slice(const SnipComplex* frame_iq,
                          uint64_t frame_n,
                          uint64_t local_start,
                          uint64_t count,
                          double sample_rate_hz,
                          DeviceBufferPool& pool,
                          cudaStream_t stream) {
  SnippetIq out;
  if (count == 0 || local_start >= frame_n) {
    return out;
  }
  const uint64_t n = std::min<uint64_t>(count, frame_n - local_start);
  auto device_out = pool.acquire(n);
  cudaMemcpyAsync(device_out.get(),
                  frame_iq + local_start,
                  n * sizeof(SnipComplex),
                  cudaMemcpyDeviceToDevice,
                  stream);
  out.device_iq = std::move(device_out);
  out.n_iq = n;
  out.sample_rate_hz = sample_rate_hz;
  return out;
}

// ---------------------------------------------------------------------------------------------
// SigMF writing (hand-rolled; no JSON dependency in this build)
// ---------------------------------------------------------------------------------------------

namespace {

std::string json_escape(const std::string& value) {
  static constexpr char hex_digits[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (ch < 0x20U) {
          escaped += "\\u00";
          escaped.push_back(hex_digits[(ch >> 4) & 0x0F]);
          escaped.push_back(hex_digits[ch & 0x0F]);
        } else {
          escaped.push_back(static_cast<char>(ch));
        }
    }
  }
  return escaped;
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open file for writing: " + path.string());
  }
  out << text;
  if (!out.good()) {
    throw std::runtime_error("failed to write file: " + path.string());
  }
}

void write_iq_file(const std::filesystem::path& path, const std::vector<SnipComplex>& iq) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open IQ file for writing: " + path.string());
  }
  // SnipComplex is two little-endian float32 (I, Q) on the host -> byte-identical to cf32_le.
  out.write(reinterpret_cast<const char*>(iq.data()),
            static_cast<std::streamsize>(iq.size() * sizeof(SnipComplex)));
  if (!out.good()) {
    throw std::runtime_error("failed to write IQ file: " + path.string());
  }
}

// One annotation to emit: its position within THIS recording (core:sample_start/count) plus the
// provenance in the ORIGINAL full-rate stream (wfgt:orig_*), so a downstream ingestor of a packed
// recording can tell which chunks belong together (same frame_number / contiguous orig range) and
// where each came from.
struct MetaAnn {
  SnipAnnotation ann;
  uint64_t file_sample_start = 0;   // within this recording's payload
  uint64_t file_sample_count = 0;
  uint64_t orig_sample_start = 0;   // original full-rate stream
  uint64_t orig_sample_count = 0;
  uint64_t frame_number = 0;
  double center_freq_hz = 0.0;
  double sample_rate_hz = 0.0;      // this snippet's own (decimated) rate
};

void append_annotation(std::ostringstream& meta, const MetaAnn& a, bool last) {
  meta << "    {\n";
  meta << "      \"core:sample_start\": " << a.file_sample_start << ",\n";
  meta << "      \"core:sample_count\": " << a.file_sample_count << ",\n";
  meta << "      \"core:freq_lower_edge\": " << a.ann.freq_lower_hz << ",\n";
  meta << "      \"core:freq_upper_edge\": " << a.ann.freq_upper_hz << ",\n";
  meta << "      \"core:label\": \"" << json_escape(a.ann.label) << "\",\n";
  meta << "      \"wfgt:kind\": \"" << json_escape(a.ann.kind) << "\",\n";
  // Original-stream provenance (which packets are together) + this chunk's own rate/center. The
  // per-annotation rate is authoritative in a container recording (mixed rates concatenated).
  meta << "      \"wfgt:frame_number\": " << a.frame_number << ",\n";
  meta << "      \"wfgt:orig_sample_start\": " << a.orig_sample_start << ",\n";
  meta << "      \"wfgt:orig_sample_count\": " << a.orig_sample_count << ",\n";
  meta << "      \"wfgt:orig_sample_end\": " << (a.orig_sample_start + a.orig_sample_count) << ",\n";
  meta << "      \"wfgt:center_frequency\": " << a.center_freq_hz << ",\n";
  // Decimation summary so the meta fully describes the processing without the IQ present:
  //   core:sample_count      = stored (decimated) sample count
  //   wfgt:orig_sample_count = full-rate sample count of the same region (above)
  //   wfgt:decimation_factor = orig / stored (1.0 = no decimation, e.g. time_only or full-band box)
  const double decim = (a.file_sample_count > 0)
      ? static_cast<double>(a.orig_sample_count) / static_cast<double>(a.file_sample_count) : 1.0;
  meta << "      \"wfgt:decimation_factor\": " << decim << ",\n";
  meta << "      \"wfgt:orig_sample_rate\": " << (a.sample_rate_hz * decim) << ",\n";
  meta << "      \"wfgt:processing\": \""
       << (decim > 1.001 ? "mix_to_baseband; lowpass to bandwidth*(1+oversample); decimate"
                         : "full-rate, no decimation (time_only, or box bandwidth ~= full span)")
       << "\",\n";
  meta << "      \"wfgt:snippet_sample_rate\": " << a.sample_rate_hz << "\n";
  meta << "    }" << (last ? "\n" : ",\n");
}

std::string build_meta(double sample_rate_hz,
                       double center_freq_hz,
                       uint64_t orig_sample_start,
                       double orig_sample_rate_hz,
                       const std::vector<MetaAnn>& anns,
                       bool container = false) {
  std::ostringstream meta;
  // Default float formatting with high precision so large Hz values (e.g. 2.4 GHz +/- kHz edges)
  // are written exactly rather than rounded to ~6 significant digits.
  meta.precision(15);
  meta << "{\n";
  meta << "  \"global\": {\n";
  meta << "    \"core:datatype\": \"cf32_le\",\n";
  meta << "    \"core:sample_rate\": " << sample_rate_hz << ",\n";
  meta << "    \"core:version\": \"1.0.0\",\n";
  meta << "    \"core:num_channels\": 1,\n";
  meta << "    \"core:description\": \"usrp_wideband signal_snipper cutout\",\n";
  if (container) {
    // Non-standard container: many variable-rate snippets concatenated end-to-end. core:sample_rate
    // above is the ORIGINAL stream rate (a reference clock); each annotation's
    // wfgt:snippet_sample_rate is the authoritative rate for its [core:sample_start,+count) slice.
    meta << "    \"wfgt:container\": true,\n";
    meta << "    \"wfgt:layout\": \"concatenated_variable_rate\",\n";
  }
  meta << "    \"wfgt:orig_sample_start\": " << orig_sample_start << ",\n";
  meta << "    \"wfgt:orig_sample_rate\": " << orig_sample_rate_hz << "\n";
  meta << "  },\n";
  meta << "  \"captures\": [\n";
  meta << "    {\n";
  meta << "      \"core:sample_start\": 0,\n";
  meta << "      \"core:frequency\": " << center_freq_hz << "\n";
  meta << "    }\n";
  meta << "  ],\n";
  meta << "  \"annotations\": [\n";
  for (size_t i = 0; i < anns.size(); ++i) {
    append_annotation(meta, anns[i], i + 1 == anns.size());
  }
  meta << "  ]\n";
  meta << "}\n";
  return meta.str();
}

}  // namespace

std::string write_sigmf_recording(const std::string& stem, const HostSnippet& snippet) {
  const std::filesystem::path data_path(stem + ".sigmf-data");
  const std::filesystem::path meta_path(stem + ".sigmf-meta");

  write_iq_file(data_path, snippet.iq);

  std::vector<MetaAnn> anns;
  for (const auto& ann : snippet.annotations) {
    MetaAnn m;
    m.ann = ann;
    m.file_sample_start = 0;
    m.file_sample_count = static_cast<uint64_t>(snippet.iq.size());
    m.orig_sample_start = snippet.orig_sample_start;
    m.orig_sample_count = snippet.orig_sample_count;
    m.frame_number = snippet.frame_number;
    m.center_freq_hz = snippet.center_freq_hz;
    m.sample_rate_hz = snippet.sample_rate_hz;
    anns.push_back(std::move(m));
  }
  const std::string meta = build_meta(snippet.sample_rate_hz,
                                      snippet.center_freq_hz,
                                      snippet.orig_sample_start,
                                      snippet.orig_sample_rate_hz,
                                      anns);
  write_text_file(meta_path, meta);
  return data_path.string();
}

std::string write_sigmf_pack(const std::string& stem, const std::vector<HostSnippet>& snippets) {
  const std::filesystem::path data_path(stem + ".sigmf-data");
  const std::filesystem::path meta_path(stem + ".sigmf-meta");

  // Concatenate IQ; every member shares the same (rate, center) group (enforced by the caller). Each
  // chunk's annotation records both its slot in this file and its original-stream provenance.
  std::vector<SnipComplex> combined;
  std::vector<MetaAnn> anns;
  double sample_rate_hz = snippets.empty() ? 0.0 : snippets.front().sample_rate_hz;
  double center_freq_hz = snippets.empty() ? 0.0 : snippets.front().center_freq_hz;
  uint64_t offset = 0;
  for (const auto& snippet : snippets) {
    combined.insert(combined.end(), snippet.iq.begin(), snippet.iq.end());
    const uint64_t count = static_cast<uint64_t>(snippet.iq.size());
    for (const auto& ann : snippet.annotations) {
      MetaAnn m;
      m.ann = ann;
      m.file_sample_start = offset;
      m.file_sample_count = count;
      m.orig_sample_start = snippet.orig_sample_start;
      m.orig_sample_count = snippet.orig_sample_count;
      m.frame_number = snippet.frame_number;
      m.center_freq_hz = snippet.center_freq_hz;
      m.sample_rate_hz = snippet.sample_rate_hz;
      anns.push_back(std::move(m));
    }
    offset += count;
  }

  write_iq_file(data_path, combined);
  const std::string meta = build_meta(sample_rate_hz, center_freq_hz,
                                      snippets.empty() ? 0 : snippets.front().orig_sample_start,
                                      sample_rate_hz, anns);
  write_text_file(meta_path, meta);
  return data_path.string();
}

std::string write_sigmf_container(const std::string& stem, const std::vector<HostSnippet>& snippets) {
  const std::filesystem::path data_path(stem + ".sigmf-data");
  const std::filesystem::path meta_path(stem + ".sigmf-meta");

  // One file holding ALL snippets concatenated end-to-end regardless of their (rate, center). Each
  // annotation self-describes its slice: core:sample_start/count locate the chunk in this file, and
  // wfgt:snippet_sample_rate / wfgt:center_frequency / freq edges give its true rate + placement.
  // A downstream ingestor slices each annotation and reads it at its own rate. This trades strict
  // SigMF conformance (one global rate) for ~one file per pack instead of one per signal.
  std::vector<SnipComplex> combined;
  std::vector<MetaAnn> anns;
  const double orig_rate = snippets.empty() ? 0.0 : snippets.front().orig_sample_rate_hz;
  const double center = snippets.empty() ? 0.0 : snippets.front().center_freq_hz;
  uint64_t offset = 0;
  for (const auto& snippet : snippets) {
    combined.insert(combined.end(), snippet.iq.begin(), snippet.iq.end());
    const uint64_t count = static_cast<uint64_t>(snippet.iq.size());
    for (const auto& ann : snippet.annotations) {
      MetaAnn m;
      m.ann = ann;
      m.file_sample_start = offset;
      m.file_sample_count = count;
      m.orig_sample_start = snippet.orig_sample_start;
      m.orig_sample_count = snippet.orig_sample_count;
      m.frame_number = snippet.frame_number;
      m.center_freq_hz = snippet.center_freq_hz;
      m.sample_rate_hz = snippet.sample_rate_hz;
      anns.push_back(std::move(m));
    }
    offset += count;
  }

  write_iq_file(data_path, combined);
  // Global core:sample_rate = the original stream rate (reference); per-annotation rates are truth.
  const std::string meta = build_meta(orig_rate, center,
                                      snippets.empty() ? 0 : snippets.front().orig_sample_start,
                                      orig_rate, anns, /*container=*/true);
  write_text_file(meta_path, meta);
  return data_path.string();
}

void write_sigmf_collection(const std::string& collection_stem,
                            const std::vector<std::string>& member_stems) {
  std::ostringstream col;
  col << "{\n";
  col << "  \"collection\": {\n";
  col << "    \"core:version\": \"1.0.0\",\n";
  col << "    \"core:streams\": [\n";
  for (size_t i = 0; i < member_stems.size(); ++i) {
    const std::string name = std::filesystem::path(member_stems[i]).filename().string();
    col << "      { \"name\": \"" << json_escape(name) << "\" }"
        << (i + 1 == member_stems.size() ? "\n" : ",\n");
  }
  col << "    ]\n";
  col << "  }\n";
  col << "}\n";
  write_text_file(std::filesystem::path(collection_stem + ".sigmf-collection"), col.str());
}

}  // namespace snip
}  // namespace holoscan::ops

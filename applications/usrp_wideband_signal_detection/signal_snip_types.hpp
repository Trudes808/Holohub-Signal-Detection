// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Message types shared between the signal_snipper operator and the sigmf_file_sink operator.
//
// Design note (the "ragged batch" pattern): once signals are snipped out of the wideband stream
// they have varying time lengths and, in frequency mode, varying sample rates. Rather than force a
// uniform tensor shape, we carry a batch of *self-describing* snippet descriptors
// (SignalSnippet) over pooled device buffers -- the same shape as NVIDIA DALI's TensorList, a
// GStreamer GstSample (buffer + caps), or an FFmpeg AVFrame that carries its own rate. Each
// SignalSnippet knows its own sample count, sample rate, and RF placement, so any downstream
// consumer (file sink, classifier, ...) can iterate the batch without a global shape assumption.

#include <cuda/std/complex>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace holoscan::ops {

using SnipComplex = cuda::std::complex<float>;  // layout-compatible with interleaved cf32 (I,Q)

// One detected-signal annotation attached to a snippet. Frequencies are absolute RF Hz so they map
// directly onto SigMF core:freq_lower_edge / core:freq_upper_edge.
struct SnipAnnotation {
  double freq_lower_hz = 0.0;
  double freq_upper_hz = 0.0;
  std::string label = "UNLABELED";  // -> core:label
  std::string kind = "waveform";    // -> wfgt:kind
};

// A single snipped signal (frequency mode) or a single time interval containing one or more signals
// (time-only mode). The IQ payload lives on the DEVICE in `device_iq` (interleaved complex float),
// so a downstream classifier subscribing to the snippets_out port consumes it directly from GPU
// memory with zero copy -- everything it needs (rate, center, dims, annotations) travels with the
// descriptor. The buffer is a pooled shared_ptr, so it fans out safely to multiple consumers (e.g. a
// classifier AND the file sink) and recycles once the last reference drops. Only the file sink ever
// copies to host; nothing here forces a device->host transfer.
struct SignalSnippet {
  // Provenance / placement in the ORIGINAL full-rate stream.
  uint64_t frame_number = 0;
  int channel = 0;
  uint64_t orig_sample_start = 0;   // full-rate global sample index of the first payload sample
  uint64_t orig_sample_count = 0;   // full-rate span covered by this snippet
  double orig_sample_rate_hz = 0.0;  // full stream rate (== span)

  // The IQ payload as stored/emitted.
  double sample_rate_hz = 0.0;   // payload rate (== orig_sample_rate_hz for time-only; decimated in freq mode)
  double center_freq_hz = 0.0;   // RF center of the payload (band center for time-only; signal center in freq mode)
  uint64_t n_iq = 0;             // number of complex samples in device_iq (varies per snippet)
  std::shared_ptr<SnipComplex> device_iq;  // device buffer (pooled); interleaved I/Q float32

  // Detected signals described by this snippet (>=1). Time-only intervals may carry several.
  std::vector<SnipAnnotation> annotations;
};

struct SnippetBatchMessage {
  std::vector<SignalSnippet> snippets;
  uint64_t frame_number = 0;
  int channel = 0;
};

// Lightweight recycling device-buffer pool so snipping does not thrash cudaMalloc/cudaFree at the
// frame rate. Buffers are bucketed by rounded-up capacity and reused. acquire() hands back a
// shared_ptr whose deleter returns the buffer to the free list, so it composes with the snippet
// lifetime (the file sink drops the last reference after writing). See the analogous recycling pool
// in operators/coherent_power_signal_detector/coherent_power_signal_detector.cu.
class DeviceBufferPool : public std::enable_shared_from_this<DeviceBufferPool> {
 public:
  // Round a request up to the next power-of-two element count so a handful of buckets cover all
  // snippet sizes and reuse stays high.
  static uint64_t bucket_for(uint64_t elements);

  // Acquire a device buffer with room for at least `elements` complex samples.
  std::shared_ptr<SnipComplex> acquire(uint64_t elements);

  ~DeviceBufferPool();

 private:
  struct Bucket {
    uint64_t capacity_elements = 0;
    std::vector<SnipComplex*> free_list;
  };
  std::mutex mutex_;
  std::vector<Bucket> buckets_;
};

}  // namespace holoscan::ops

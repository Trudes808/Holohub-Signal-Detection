// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Reusable, pipeline-agnostic core for snipping detected signals out of the wideband stream.
// These free functions are shared by the signal_snipper operator (live/replay graph) and can be
// called directly from batch/offline tooling. They contain no Holoscan operator state.

#include "signal_snip_types.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::ops::snip {

// --- Clustering ------------------------------------------------------------------------------

// Axis-aligned bounding box of a connected component in mask pixel coordinates.
// Rows are time (top->bottom), cols are frequency (left->right); ranges are inclusive.
struct BoundingBox {
  int row0 = 0;
  int row1 = 0;
  int col0 = 0;
  int col1 = 0;
  int pixel_count = 0;
};

// 4-connected connected-component labeling over a binary (0/non-zero) mask, returning one bounding
// box per component with at least `min_pixels` pixels. Adapted from the detector's
// label_mask_connected_components (coherent_power_signal_detector.cu).
std::vector<BoundingBox> label_components(const std::vector<uint8_t>& mask,
                                          int rows,
                                          int cols,
                                          int min_pixels);

// Coalesce boxes whose bounding boxes are within `gap_rows` / `gap_cols` of each other (in mask
// pixels), summing pixel counts. Merges fragments of one signal split by small mask gaps; keeps
// distinct signals separated by more than the gap apart. Iterates to a fixed point.
std::vector<BoundingBox> merge_boxes(std::vector<BoundingBox> boxes, int gap_rows, int gap_cols);

// Geometry needed to map mask pixels back to physical (sample / Hz) coordinates for one frame.
struct FrameGeometry {
  int mask_rows = 0;               // mask height (time)
  int mask_cols = 0;               // mask width (frequency)
  uint64_t frame_sample_start = 0;  // full-rate global sample index of the frame's first sample
  uint64_t frame_sample_count = 0;  // full-rate samples represented by the whole mask
  double sample_rate_hz = 0.0;      // full stream rate (== span)
  double center_freq_hz = 0.0;      // RF center of the band
};

// Physical placement of one detected component within a frame.
struct PhysicalRegion {
  uint64_t sample_start = 0;   // full-rate global sample index
  uint64_t sample_count = 0;   // full-rate span
  uint64_t local_start = 0;    // offset within the frame's IQ buffer
  double freq_lower_hz = 0.0;  // absolute RF Hz
  double freq_upper_hz = 0.0;
  double freq_center_hz = 0.0;
  double bandwidth_hz = 0.0;
  double sample_rate_hz = 0.0;  // full stream rate (carried through for decimation math)
  double center_freq_hz = 0.0;  // RF center of the band (for baseband offset)
};

PhysicalRegion map_box_to_physical(const BoundingBox& box, const FrameGeometry& geom);

// --- Digital down-conversion ----------------------------------------------------------------

struct SnipDspParams {
  double oversample_percent = 25.0;   // extra bandwidth margin kept beyond the detected bandwidth
  bool enable_downsample = true;      // if false, keep full rate (decimation factor 1)
  double bandwidth_margin_hz = 0.0;   // absolute Hz added to detected bandwidth before filtering
  int fir_num_taps = 129;             // low-pass FIR length (odd)
};

// Result of a single extraction: a device buffer plus its realized rate/length.
struct SnippetIq {
  std::shared_ptr<SnipComplex> device_iq;
  uint64_t n_iq = 0;
  double sample_rate_hz = 0.0;
};

// Frequency mode: mix `region` to baseband, low-pass to its bandwidth (+ oversample margin), and
// decimate. `frame_iq` points to the frame's contiguous full-rate IQ on the device; `frame_n` is
// its length in complex samples. Runs on `stream`; buffers come from `pool`.
SnippetIq ddc_extract(const SnipComplex* frame_iq,
                      uint64_t frame_n,
                      const PhysicalRegion& region,
                      const SnipDspParams& params,
                      DeviceBufferPool& pool,
                      cudaStream_t stream);

// Time-only mode: copy the full-band, full-rate IQ for [local_start, local_start+count) into a
// pooled buffer. Runs on `stream`.
SnippetIq copy_time_slice(const SnipComplex* frame_iq,
                          uint64_t frame_n,
                          uint64_t local_start,
                          uint64_t count,
                          double sample_rate_hz,
                          DeviceBufferPool& pool,
                          cudaStream_t stream);

// --- SigMF writing --------------------------------------------------------------------------

// Copy of a snippet's IQ already staged on the host, plus its descriptors, for writing.
struct HostSnippet {
  std::vector<SnipComplex> iq;  // interleaved cf32 (I,Q) on host
  double sample_rate_hz = 0.0;
  double center_freq_hz = 0.0;
  uint64_t orig_sample_start = 0;
  double orig_sample_rate_hz = 0.0;
  uint64_t frame_number = 0;
  int channel = 0;
  std::vector<SnipAnnotation> annotations;
};

// Write one SigMF recording (.sigmf-data + .sigmf-meta) at `stem` (no extension). datatype cf32_le.
// Returns the .sigmf-data path. Throws on I/O error.
std::string write_sigmf_recording(const std::string& stem, const HostSnippet& snippet);

// Write a SigMF Collection (.sigmf-collection) referencing already-written recordings (by stem).
// Used to tie together a "pack" whose members have heterogeneous sample rates.
void write_sigmf_collection(const std::string& collection_stem,
                            const std::vector<std::string>& member_stems);

// Write a single uniform-rate SigMF recording that concatenates several host snippets that share a
// sample rate, emitting one annotation per contained signal at the right sample offset. Returns the
// .sigmf-data path.
std::string write_sigmf_pack(const std::string& stem, const std::vector<HostSnippet>& snippets);

}  // namespace holoscan::ops::snip

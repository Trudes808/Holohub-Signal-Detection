// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace holoscan::ops {

// Off-data-path per-frame mask dumper used ONLY for DINO-FT validation (offline-vs-loopback A/B).
//
// Design: default-inert. configure() with a non-empty dir spins up a single background writer
// thread; submit() copies the host mask into a bounded queue and returns immediately, so the
// detector's compute() never blocks on disk I/O -- essential here because we are specifically
// measuring the real-time ingest, and a synchronous write would perturb the packet drops we want
// to observe. When disabled (empty dir) every method is a no-op and there is zero hot-path cost.
//
// Output matches the offline eval's mask_arrays format so the existing compare tooling
// (infocom_evals/.../mask_eval_metrics.py) reads both trees identically:
//   - NumPy v1.0 .npy, dtype '|u1' (uint8 0/1), C-order, shape (rows, cols)
//   - a manifest CSV: seq,channel,frame_number,rows,cols,mask_npy
class MaskDumpWriter {
 public:
  MaskDumpWriter();
  ~MaskDumpWriter();

  MaskDumpWriter(const MaskDumpWriter&) = delete;
  MaskDumpWriter& operator=(const MaskDumpWriter&) = delete;

  // dir: a host-mounted container path (e.g. /workspace/spectrograms/...); created if missing.
  // max_frames: total frames to write across all channels; <=0 means unlimited.
  void configure(const std::string& dir, int max_frames);

  bool enabled() const { return enabled_; }

  // False once the frame budget is reached; the caller can then skip the D2H copy entirely.
  bool wants_more() const;

  // Enqueue a COPY of host_mask (rows*cols uint8, C-order). Non-blocking: if the writer has
  // fallen behind and the queue is full, the frame is dropped and counted (logged at stop) so
  // the data path is never stalled. If host_spec != nullptr, ALSO dumps the rows*cols float32
  // input spectrogram (the [0,1] image the model saw) as spec_*.npy for mask-on-spectrogram overlays.
  void submit(int channel, uint64_t frame_number, int rows, int cols, const uint8_t* host_mask,
              const float* host_spec = nullptr);

  // Drain the queue, join the writer, close the manifest, and log a one-line summary.
  void flush_and_stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool enabled_ = false;
};

}  // namespace holoscan::ops

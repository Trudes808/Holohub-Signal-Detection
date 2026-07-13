// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "signal_snip_core.hpp"
#include "signal_snip_types.hpp"

#include <holoscan/holoscan.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace holoscan::ops {

// Writes snipped signals to disk in SigMF (cf32_le). Two modes:
//   - "per_signal": one .sigmf-data + .sigmf-meta per snippet.
//   - "pack": accumulate snippets over N mask-frames, then flush. Snippets are grouped by
//     (rate, center) into concatenated recordings, tied by a .sigmf-collection when a pack spans
//     more than one group.
//
// Two-stage, device-memory-friendly path:
//   1. compute() copies each snippet device->host into CPU memory RIGHT AWAY (fast, ~GB/s), then
//      hands the host-staged batch to a bounded queue. The device IQ (pooled) is released back to
//      the snipper's pool at the end of compute(), so device memory recycles immediately and never
//      backs up behind slow disk.
//   2. A BACKGROUND CPU THREAD drains the queue and does only host-memory -> file writes.
// compute() never blocks on disk, so it can't back up the upstream transmitter (which otherwise
// overflows fatally as GXF_EXCEEDING_PREALLOCATED_SIZE under the event-based scheduler). If the
// writer falls behind, compute() drops whole batches with a loud OVERFLOW (bounding HOST memory).
//
// This sink is the ONLY place that touches host memory / files. The emitted SnippetBatchMessage
// stays fully device-resident so a downstream classifier can consume it in-memory with no copy.
class SigmfFileSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SigmfFileSinkOp)

  SigmfFileSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  void writer_loop();
  snip::HostSnippet stage_to_host(const SignalSnippet& snippet) const;  // compute thread (D2H)
  void write_host_batch(const std::vector<snip::HostSnippet>& snippets);  // writer thread (host->file)
  void flush_pack();  // writer thread only

  holoscan::Parameter<std::string> mode_;
  holoscan::Parameter<int> pack_frames_;
  holoscan::Parameter<std::string> output_dir_;
  holoscan::Parameter<std::string> filename_prefix_;
  holoscan::Parameter<int> max_queued_batches_;

  // Producer/consumer handoff to the background writer: HOST-staged batches (device already freed).
  std::thread writer_thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::vector<snip::HostSnippet>> queue_;
  bool stopping_ = false;

  // Writer-thread-only state.
  std::vector<snip::HostSnippet> pending_;  // accumulated pack
  int frames_in_pack_ = 0;                  // distinct frames accumulated (not batches)
  uint64_t last_pack_frame_ = 0;
  bool have_last_pack_frame_ = false;
  uint64_t pack_index_ = 0;
  uint64_t files_written_ = 0;
  uint64_t batches_dropped_ = 0;
  uint64_t snippets_dropped_ = 0;   // signals dropped under real-time overflow
  uint64_t samples_dropped_ = 0;    // payload IQ samples dropped
  uint64_t orig_samples_dropped_ = 0;  // original full-rate samples dropped
};

}  // namespace holoscan::ops

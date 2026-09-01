// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// SnippetCompressionOp: real-time IQ compression on the data-saved path.
//
// Sits between signal_snipper (snippets_out) and sigmf_file_sink: consumes a SnippetBatchMessage,
// compresses each snippet's device-resident cf32 payload on the GPU, and emits the SAME message
// type with the compression fields filled (codec/comp_bytes/device_comp/...) and the fat cf32
// buffer released. Detector- and snip-mode-agnostic by construction: everything it needs travels
// in the self-describing SignalSnippet descriptor.
//
// v1 codecs (all decode-grade at the demo's SNR ladder):
//   none  - passthrough (message forwarded untouched; zero cost)
//   sc16  - per-snippet peak-scaled int16 I/Q. Exactly 2.0x vs cf32; ~96 dB quantization SNR.
//           (The radio wire format IS sc16 -- cf32 files were 2x inflated from the start.)
//   bfp12 - O-RAN-style block floating point: per-block (bfp_block complex samples) shared int8
//           power-of-two exponent + 12-bit two's-complement mantissas packed 2-per-3-bytes.
//           ~2.65x vs cf32; ~66 dB block SNR.
//   bfp8  - same layout with int8 mantissas. ~3.97x vs cf32; ~42 dB block SNR (still above the
//           demo's decode floor, but the first codec where the 50 dB "clean" rung notices).
//
// Codec is a config param, and optionally LIVE-SWITCHABLE: when control_json_path is set the
// operator re-reads that JSON (the DEMO CONTROLS file) and adopts its "codec" value without an
// app restart -- same pattern the visualizer uses for gate/snr/detector.
//
// Entropy stages (nvCOMP LZ4/Zstd) and ZFP fixed-rate slot in behind the same codec param once
// their libraries land in the container (see notes/compression plan).

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include "signal_snip_types.hpp"

namespace holoscan::ops {

class SnippetCompressionOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SnippetCompressionOp)
  SnippetCompressionOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;
  void stop() override;

 private:
  // Re-read control_json_path (if set) at most every ~0.5 s and adopt its "codec" value.
  void maybe_refresh_codec();
  // Grow the per-batch max-abs scratch arrays (device + pinned host) to hold `n` snippets.
  void ensure_max_scratch(size_t n);

  Parameter<std::string> codec_;
  Parameter<int> bfp_block_;
  Parameter<std::string> control_json_path_;

  std::string active_codec_;          // runtime codec (starts as codec_, control file may switch it)
  cudaStream_t stream_ = nullptr;
  std::shared_ptr<DeviceBufferPool> pool_;

  unsigned int* d_max_bits_ = nullptr;  // per-snippet max|x| as float bits (device)
  unsigned int* h_max_bits_ = nullptr;  // pinned mirror
  size_t max_scratch_capacity_ = 0;

  // Control-file polling state.
  double last_control_check_s_ = 0.0;
  int64_t control_mtime_ns_ = -1;

  // Aggregate economics (logged periodically).
  uint64_t logical_bytes_ = 0;
  uint64_t stored_bytes_ = 0;
  uint64_t snippets_compressed_ = 0;
  double last_log_s_ = 0.0;
};

}  // namespace holoscan::ops

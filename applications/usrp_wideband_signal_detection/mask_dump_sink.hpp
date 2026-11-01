// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Detector-agnostic emitted-mask dump (VALIDATION ONLY; default off).
//
// Fans out from the active detector's "mask_out" next to the visualizer and the signal snipper and
// records every DetectorMaskMessage exactly as those consumers see it (same grid, same
// frame_number) as NumPy .npy files plus a manifest CSV, via the background MaskDumpWriter the
// fine-tuned DINO operator already uses for its own A/B dump. Per frame the manifest also carries
// the host receive time, the detector's compute time (metadata detector_inference_ms from the
// fine-tuned DINO operator, coherent_timing_total_ms from the coherent-power detector) and the CHDR
// ingest fields (batch index, packets assembled / expected, partial flag), so a live or loopback run
// can be scored against the offline eval AND its cadence, latency and drop pattern reconstructed
// from the manifest alone.
//
// Enabled by main.cpp when `mask_dump.dir` (config) or USRP_MASK_DUMP_DIR (env, wins) is set; the
// D2H copy runs on a private non-blocking stream so the legacy default stream is never implicitly
// synchronized, and the writer never blocks the data path (it drops + counts if its queue fills).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>

#include <finetuned_dino_mask_dump.hpp>
#include "spectrogram_visualization.hpp"

namespace holoscan::ops {

class MaskDumpSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MaskDumpSinkOp)

  MaskDumpSinkOp() = default;
  ~MaskDumpSinkOp() override {
    if (copy_stream_ != nullptr) {
      cudaStreamDestroy(copy_stream_);
      copy_stream_ = nullptr;
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.input<DetectorMaskMessage>("in");
    spec.param(dir_, "dir", "Dump directory",
               "Container path for mask_*.npy + mask_dump_manifest.csv; \"\" disables.",
               std::string(""));
    spec.param(max_frames_, "max_frames", "Max frames",
               "Total masks to write before the dump goes quiet; 0 = unlimited.", 0);
  }

  void initialize() override {
    Operator::initialize();
    writer_ = std::make_shared<MaskDumpWriter>();
    writer_->configure(dir_.get(), max_frames_.get());
    if (cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking) != cudaSuccess) {
      copy_stream_ = nullptr;  // fall back to the synchronous legacy-stream copy
    }
  }

  void stop() override {
    if (writer_) writer_->flush_and_stop();
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto maybe = op_input.receive<DetectorMaskMessage>("in");
    if (!maybe) return;
    const auto& mask = *maybe;
    const uint64_t host_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    ++received_;
    if (!writer_ || !writer_->wants_more()) return;
    if (mask.width <= 0 || mask.height <= 0) return;

    const size_t n = static_cast<size_t>(mask.width) * static_cast<size_t>(mask.height);
    if (host_buf_.size() < n) host_buf_.resize(n);
    if (mask.device_pixels) {
      // Producers synchronize before emitting (consumers assume a complete mask on receipt), so a
      // plain copy on a private stream is ordered correctly and avoids the legacy-stream sync.
      cudaError_t err;
      if (copy_stream_ != nullptr) {
        err = cudaMemcpyAsync(host_buf_.data(), mask.device_pixels.get(), n, cudaMemcpyDeviceToHost,
                              copy_stream_);
        if (err == cudaSuccess) err = cudaStreamSynchronize(copy_stream_);
      } else {
        err = cudaMemcpy(host_buf_.data(), mask.device_pixels.get(), n, cudaMemcpyDeviceToHost);
      }
      if (err != cudaSuccess) {
        if ((copy_failures_++ % 64) == 0) {
          HOLOSCAN_LOG_WARN("mask_dump_sink: D2H copy failed ({}); frame {} skipped",
                            cudaGetErrorString(err), mask.frame_number);
        }
        return;
      }
    } else if (mask.pixels.size() >= n) {
      std::copy(mask.pixels.begin(), mask.pixels.begin() + static_cast<std::ptrdiff_t>(n),
                host_buf_.begin());
    } else {
      return;
    }

    MaskDumpFrameInfo info;
    info.host_mono_ns = host_ns;
    auto meta = metadata();
    if (meta) {
      if (meta->has_key("detector_inference_ms")) {
        info.detector_ms = meta->get<double>("detector_inference_ms", -1.0);
      } else if (meta->has_key("coherent_timing_total_ms")) {
        info.detector_ms = meta->get<double>("coherent_timing_total_ms", -1.0);
      }
      info.chdr_batch_index = meta->get<uint64_t>("chdr_batch_index", 0ull);
      info.chdr_pkts = meta->get<uint32_t>("chdr_packets_in_batch", 0u);
      info.chdr_expected_pkts = meta->get<uint32_t>("chdr_expected_packets_in_batch", 0u);
      info.chdr_partial = meta->get<bool>("chdr_partial_batch", false);
    }
    writer_->submit(mask.channel, mask.frame_number, mask.height, mask.width, host_buf_.data(),
                    nullptr, &info);
  }

 private:
  Parameter<std::string> dir_;
  Parameter<int> max_frames_;
  std::shared_ptr<MaskDumpWriter> writer_;
  std::vector<uint8_t> host_buf_;
  cudaStream_t copy_stream_ = nullptr;
  uint64_t received_ = 0;
  uint64_t copy_failures_ = 0;
};

}  // namespace holoscan::ops

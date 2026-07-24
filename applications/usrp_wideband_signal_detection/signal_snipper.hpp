// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "signal_snip_core.hpp"
#include "signal_snip_types.hpp"
#include "spectrogram_visualization.hpp"  // DetectorMaskMessage

#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <holoscan/holoscan.hpp>
#include <matx.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

// Ingests detector masks and the raw time-domain IQ stream (tapped from the CHDR converter / offline
// source, upstream of the FFT), clusters mask pixels into per-signal bounding boxes, and cuts the
// corresponding IQ out of the stream. Emits a SnippetBatchMessage of self-describing signal cutouts.
//
// Two config modes:
//   - "time_only": keep the time regions that contain any signal (full bandwidth, full rate); each
//     emitted snippet carries an annotation per signal detected within that interval.
//   - "frequency": additionally isolate each signal -- digital down-convert to baseband, low-pass to
//     the detected bandwidth (+ oversample margin), and optionally decimate to the minimum rate.
//
// Correlation: the FFT stamps fft_emitted_frame_number = its per-input counter, which the detectors
// copy into DetectorMaskMessage.frame_number. That counter equals the CHDR arrival index, so this
// operator keys a small device IQ ring by its own IQ arrival counter and matches masks directly.
class SignalSnipperOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SignalSnipperOp)

  using iq_in_t = std::tuple<matx::tensor_t<cuda::std::complex<float>, 2>, cudaStream_t>;

  SignalSnipperOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  struct RingEntry {
    uint64_t frame_number = 0;
    std::shared_ptr<SnipComplex> device_iq;
    uint64_t n_iq = 0;
  };

  void ingest_iq(holoscan::InputContext& op_input);
  RingEntry* find_ring_entry(uint64_t frame_number);
  // Snip one mask's signals and APPEND them to `batch` (no emit). All masks drained in a single
  // compute() accumulate into one batch that is emitted exactly once -- Holoscan's transmitter
  // stages one message per compute tick, so emitting per-mask would overflow it fatally.
  void process_mask(const DetectorMaskMessage& mask, SnippetBatchMessage& batch);
  void prune_ring();

  // Parameters.
  holoscan::Parameter<std::string> mode_;
  holoscan::Parameter<double> oversample_percent_;
  holoscan::Parameter<bool> enable_downsample_;
  holoscan::Parameter<double> bandwidth_margin_hz_;
  holoscan::Parameter<int> min_box_pixels_;
  holoscan::Parameter<double> min_mask_bandwidth_hz_;  // pre-label mask run-length filter (0 = disabled)
  holoscan::Parameter<double> min_bandwidth_hz_;   // post-merge min box bandwidth (0 = disabled)
  holoscan::Parameter<double> min_duration_s_;     // post-merge min box duration  (0 = disabled)
  holoscan::Parameter<int> merge_gap_rows_;
  holoscan::Parameter<int> merge_gap_cols_;
  holoscan::Parameter<int> fir_num_taps_;
  holoscan::Parameter<int> ring_depth_;
  holoscan::Parameter<int> channel_filter_;
  holoscan::Parameter<double> center_frequency_hz_;  // fallback if metadata absent
  holoscan::Parameter<double> sample_rate_hz_;        // fallback if metadata absent

  // State.
  std::shared_ptr<DeviceBufferPool> pool_;
  // Per-frame host scratch, reused to avoid reallocating ~5 MB buffers every masked frame.
  std::vector<uint8_t> host_mask_;
  snip::CcScratch cc_scratch_;
  std::deque<RingEntry> ring_;
  uint64_t iq_arrival_counter_ = 0;
  uint64_t last_processed_mask_frame_ = 0;
  cudaStream_t snip_stream_ = nullptr;
  std::vector<cudaEvent_t> event_pool_;  // reused round-robin, sized to ring depth
  uint64_t masks_processed_ = 0;
  uint64_t snippets_emitted_ = 0;
};

}  // namespace holoscan::ops

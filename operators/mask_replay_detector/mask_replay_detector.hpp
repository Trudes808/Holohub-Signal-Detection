// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::ops {

// Replays PRECOMPUTED detector masks (written by the offline eval as
// mask_ch{c}_f{N}_{H}x{W}.npy) back into the pipeline so the C++ signal_snipper
// can snip ANY detector's masks -- including the Python-only fine-tuned models --
// without that detector existing as a C++ operator.
//
// It is a drop-in "detector": same ports as CudaDinoDetector
//   in       = the spectrogram tuple (used ONLY for message metadata:
//              fft_emitted_frame_number + offline_source_* IQ offsets; the tensor
//              payload is ignored)
//   mask_out = DetectorMaskMessage (host pixels loaded from the .npy)
// so it slots into add_flow(spectrogram, detector) -> add_flow(detector, snipper)
// with no compose change beyond one DetectorAdapter entry.
class MaskReplayDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MaskReplayDetector)

  MaskReplayDetector() = default;
  ~MaskReplayDetector() override = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> mask_dir_;
  holoscan::Parameter<int> channel_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<int> num_channels_;

  uint64_t frame_counter_ = 0;      // fallback index when metadata lacks fft_emitted_frame_number
  bool startup_log_emitted_ = false;
  int last_rows_ = 0;               // cached geometry for zero-fill on a missing frame
  int last_cols_ = 0;
  uint64_t missing_mask_count_ = 0;
};

}  // namespace holoscan::ops

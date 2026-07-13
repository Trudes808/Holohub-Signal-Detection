// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "signal_snip_core.hpp"
#include "signal_snip_types.hpp"

#include <holoscan/holoscan.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace holoscan::ops {

// Writes snipped signals to disk in SigMF (cf32_le). Two modes:
//   - "per_signal": one .sigmf-data + .sigmf-meta per snippet, written as they arrive.
//   - "pack": accumulate snippets over N mask-frames, then flush. Snippets are grouped by sample
//     rate; each rate group becomes one concatenated recording. If a pack spans more than one rate,
//     the group recordings are tied together with a .sigmf-collection (the SigMF-native way to carry
//     a heterogeneous-rate set, analogous to a multi-resolution recording collection).
class SigmfFileSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SigmfFileSinkOp)

  SigmfFileSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;
  void stop() override;

 private:
  snip::HostSnippet stage_to_host(const SignalSnippet& snippet) const;
  void flush_pack();

  holoscan::Parameter<std::string> mode_;
  holoscan::Parameter<int> pack_frames_;
  holoscan::Parameter<std::string> output_dir_;
  holoscan::Parameter<std::string> filename_prefix_;

  std::vector<snip::HostSnippet> pending_;  // accumulated pack (host-staged)
  int frames_in_pack_ = 0;
  uint64_t pack_index_ = 0;
  uint64_t files_written_ = 0;
};

}  // namespace holoscan::ops

// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "sigmf_file_sink.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::ops {

void SigmfFileSinkOp::setup(OperatorSpec& spec) {
  auto& input_port = spec.input<SnippetBatchMessage>("in", holoscan::IOSpec::IOSize{8});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));

  spec.param(mode_, "mode", "Mode", "'per_signal' or 'pack'.", std::string("per_signal"));
  spec.param(pack_frames_, "pack_frames", "Pack frames",
             "In pack mode, flush every N mask-frames of snippets.", 16);
  spec.param(output_dir_, "output_dir", "Output directory", "Directory for emitted SigMF files.",
             std::string("/tmp/usrp_spectrograms/snippets"));
  spec.param(filename_prefix_, "filename_prefix", "Filename prefix", "Prefix for emitted files.",
             std::string("snip"));
}

snip::HostSnippet SigmfFileSinkOp::stage_to_host(const SignalSnippet& snippet) const {
  snip::HostSnippet host;
  host.iq.resize(snippet.n_iq);
  if (snippet.n_iq > 0 && snippet.device_iq) {
    const cudaError_t status = cudaMemcpy(host.iq.data(),
                                          snippet.device_iq.get(),
                                          snippet.n_iq * sizeof(SnipComplex),
                                          cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("sigmf_file_sink: cudaMemcpy failed: ") +
                               cudaGetErrorString(status));
    }
  }
  host.sample_rate_hz = snippet.sample_rate_hz;
  host.center_freq_hz = snippet.center_freq_hz;
  host.orig_sample_start = snippet.orig_sample_start;
  host.orig_sample_rate_hz = snippet.orig_sample_rate_hz;
  host.frame_number = snippet.frame_number;
  host.channel = snippet.channel;
  host.annotations = snippet.annotations;
  return host;
}

namespace {

std::string stem_for(const std::string& dir,
                     const std::string& prefix,
                     const snip::HostSnippet& s,
                     uint64_t seq) {
  std::ostringstream name;
  name << prefix << "_ch" << s.channel << "_f" << s.frame_number << "_s" << seq;
  return (std::filesystem::path(dir) / name.str()).string();
}

// Group by rounded sample rate so that snippets sharing a rate can be concatenated into one
// recording. Integer Hz rounding is enough to separate distinct decimation factors.
int64_t rate_key(double rate_hz) { return static_cast<int64_t>(std::llround(rate_hz)); }

}  // namespace

void SigmfFileSinkOp::flush_pack() {
  if (pending_.empty()) {
    frames_in_pack_ = 0;
    return;
  }

  const uint64_t pack_id = pack_index_++;
  std::ostringstream pack_name;
  pack_name << filename_prefix_.get() << "_pack" << pack_id;
  const std::string pack_stem = (std::filesystem::path(output_dir_.get()) / pack_name.str()).string();

  // Preserve arrival order within each rate group.
  std::map<int64_t, std::vector<snip::HostSnippet>> groups;
  std::vector<int64_t> group_order;
  for (auto& snippet : pending_) {
    const int64_t key = rate_key(snippet.sample_rate_hz);
    if (groups.find(key) == groups.end()) {
      group_order.push_back(key);
    }
    groups[key].push_back(std::move(snippet));
  }

  std::vector<std::string> member_stems;
  int group_index = 0;
  for (const int64_t key : group_order) {
    std::ostringstream member_name;
    member_name << pack_name.str() << "_r" << group_index++;
    const std::string member_stem =
        (std::filesystem::path(output_dir_.get()) / member_name.str()).string();
    snip::write_sigmf_pack(member_stem, groups[key]);
    member_stems.push_back(member_stem);
    ++files_written_;
  }

  // Mixed-rate pack: tie the per-rate recordings together with a SigMF Collection.
  if (member_stems.size() > 1) {
    snip::write_sigmf_collection(pack_stem, member_stems);
  }

  HOLOSCAN_LOG_INFO("sigmf_file_sink: flushed pack {} ({} rate group(s), {} recording(s)).",
                    pack_id, member_stems.size(), member_stems.size());

  pending_.clear();
  frames_in_pack_ = 0;
}

void SigmfFileSinkOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto in = op_input.receive<SnippetBatchMessage>("in");
  if (!in) {
    return;
  }
  const SnippetBatchMessage batch = std::move(in.value());

  std::filesystem::create_directories(output_dir_.get());

  if (mode_.get() == "pack") {
    for (const auto& snippet : batch.snippets) {
      pending_.push_back(stage_to_host(snippet));
    }
    ++frames_in_pack_;
    if (frames_in_pack_ >= std::max(1, pack_frames_.get())) {
      flush_pack();
    }
    return;
  }

  // per_signal: write each snippet immediately.
  uint64_t seq = 0;
  for (const auto& snippet : batch.snippets) {
    const snip::HostSnippet host = stage_to_host(snippet);
    const std::string stem = stem_for(output_dir_.get(), filename_prefix_.get(), host, seq++);
    snip::write_sigmf_recording(stem, host);
    ++files_written_;
  }
  if (!batch.snippets.empty()) {
    HOLOSCAN_LOG_INFO("sigmf_file_sink: wrote {} recording(s) for frame {}.",
                      batch.snippets.size(), batch.frame_number);
  }
}

void SigmfFileSinkOp::stop() {
  if (!pending_.empty()) {
    flush_pack();
  }
  HOLOSCAN_LOG_INFO("sigmf_file_sink: total files written {}.", files_written_);
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

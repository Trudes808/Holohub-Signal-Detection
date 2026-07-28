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
#include <system_error>
#include <string>
#include <utility>
#include <vector>

namespace holoscan::ops {

void SigmfFileSinkOp::setup(OperatorSpec& spec) {
  auto& input_port = spec.input<SnippetBatchMessage>("in", holoscan::IOSpec::IOSize{16});
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
  spec.param(write_iq_, "write_iq", "Write IQ",
             "If false, write only the .sigmf-meta and immediately delete the .sigmf-data. The snip "
             "footprint stays exactly measurable (sample_count/rate in the meta) without storing IQ.",
             true);
  spec.param(max_queued_batches_, "max_queued_batches", "Max queued batches",
             "Bound on batches awaiting the background writer; excess is dropped so the pipeline "
             "never blocks on disk. Each queued batch pins its device IQ until written, so this also "
             "caps extra device memory.", 16);
}

void SigmfFileSinkOp::initialize() {
  holoscan::Operator::initialize();
  stopping_ = false;
  writer_thread_ = std::thread([this] { writer_loop(); });
}

// --- Background writer -----------------------------------------------------------------------

void SigmfFileSinkOp::writer_loop() {
  while (true) {
    std::vector<snip::HostSnippet> batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        if (stopping_) {
          break;
        }
        continue;
      }
      batch = std::move(queue_.front());
      queue_.pop_front();
    }
    try {
      write_host_batch(batch);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("sigmf_file_sink writer: {}", e.what());
    }
  }
  // Flush any partial pack accumulated before shutdown.
  if (!pending_.empty()) {
    try {
      flush_pack();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("sigmf_file_sink writer flush: {}", e.what());
    }
  }
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
  host.orig_sample_count = snippet.orig_sample_count;
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

int64_t rate_key(double hz) { return static_cast<int64_t>(std::llround(hz)); }

}  // namespace

void SigmfFileSinkOp::flush_pack() {
  if (pending_.empty()) {
    frames_in_pack_ = 0;
    have_last_pack_frame_ = false;
    return;
  }

  const uint64_t pack_id = pack_index_++;
  std::ostringstream pack_name;
  pack_name << filename_prefix_.get() << "_pack" << pack_id;
  const std::string pack_stem = (std::filesystem::path(output_dir_.get()) / pack_name.str()).string();

  // Do all snippets share one (rate, center)? If so (typically time_only mode) write one standard
  // SigMF recording. Otherwise (typically frequency mode: every signal a distinct rate/center) write
  // ONE container recording holding them all -- one file per pack instead of one per signal, which
  // is what makes dense / low-emit_stride capture sustainable.
  bool uniform = true;
  const std::pair<int64_t, int64_t> first_key{rate_key(pending_.front().sample_rate_hz),
                                              rate_key(pending_.front().center_freq_hz)};
  for (const auto& snippet : pending_) {
    if (rate_key(snippet.sample_rate_hz) != first_key.first ||
        rate_key(snippet.center_freq_hz) != first_key.second) {
      uniform = false;
      break;
    }
  }

  if (uniform) {
    const std::string dpath = snip::write_sigmf_pack(pack_stem, pending_);
    if (!write_iq_.get()) { std::error_code ec; std::filesystem::remove(dpath, ec); }
    ++files_written_;
    HOLOSCAN_LOG_INFO("sigmf_file_sink: flushed pack {} ({} snippet(s), 1 uniform recording).",
                      pack_id, pending_.size());
  } else {
    const std::string dpath = snip::write_sigmf_container(pack_stem, pending_);
    if (!write_iq_.get()) { std::error_code ec; std::filesystem::remove(dpath, ec); }
    ++files_written_;
    HOLOSCAN_LOG_INFO("sigmf_file_sink: flushed pack {} ({} snippet(s), 1 variable-rate container).",
                      pack_id, pending_.size());
  }

  pending_.clear();
  frames_in_pack_ = 0;
  have_last_pack_frame_ = false;
}

void SigmfFileSinkOp::write_host_batch(const std::vector<snip::HostSnippet>& snippets) {
  // Host-only: the batch's IQ is already in CPU memory (staged in compute). Just write files.
  if (mode_.get() == "pack") {
    // Count DISTINCT original frames (a single batch may merge several masks/frames), so a pack
    // covers pack_frames real frames regardless of how compute batched them.
    for (const auto& host : snippets) {
      if (!have_last_pack_frame_ || host.frame_number != last_pack_frame_) {
        ++frames_in_pack_;
        last_pack_frame_ = host.frame_number;
        have_last_pack_frame_ = true;
      }
      pending_.push_back(host);
    }
    if (frames_in_pack_ >= std::max(1, pack_frames_.get())) {
      flush_pack();
    }
    return;
  }

  // per_signal: write each snippet immediately.
  uint64_t seq = 0;
  for (const auto& host : snippets) {
    const std::string stem = stem_for(output_dir_.get(), filename_prefix_.get(), host, seq++);
    const std::string dpath = snip::write_sigmf_recording(stem, host);
    if (!write_iq_.get()) { std::error_code ec; std::filesystem::remove(dpath, ec); }
    ++files_written_;
  }
  if (!snippets.empty()) {
    HOLOSCAN_LOG_INFO("sigmf_file_sink: wrote {} recording(s) for frame {}.",
                      snippets.size(), snippets.front().frame_number);
  }
}

// --- Operator compute: stage device->host FAST, hand host batch to the async writer -----------

void SigmfFileSinkOp::compute(InputContext& op_input, OutputContext&, ExecutionContext&) {
  auto in = op_input.receive<SnippetBatchMessage>("in");
  if (!in) {
    return;
  }
  const SnippetBatchMessage batch = std::move(in.value());

  // Tally before doing work, so an overflow reports exactly what is lost.
  const uint64_t frame_number = batch.frame_number;
  const uint64_t batch_signals = static_cast<uint64_t>(batch.snippets.size());
  uint64_t batch_samples = 0;
  uint64_t batch_orig_samples = 0;
  for (const auto& snippet : batch.snippets) {
    batch_samples += snippet.n_iq;
    batch_orig_samples += snippet.orig_sample_count;
  }

  // Drop-check FIRST (bounds host-queue memory) so we don't waste the device->host copy on a batch
  // we'd only discard. compute() is serialized per operator, and the writer only shrinks the queue,
  // so "not full" stays valid until we push below. Either way `batch` drops at end of compute and
  // its pooled device IQ recycles immediately.
  bool dropped;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    dropped = static_cast<int>(queue_.size()) >= std::max(1, max_queued_batches_.get());
    if (dropped) {
      ++batches_dropped_;
      snippets_dropped_ += batch_signals;
      samples_dropped_ += batch_samples;
      orig_samples_dropped_ += batch_orig_samples;
    }
  }

  if (dropped) {
    // Best-effort: never block the pipeline on disk. Loudly report what real-time pressure cost.
    // Log the onset (first drop) and then periodically, always with running totals.
    if (batches_dropped_ == 1 || (batches_dropped_ % 16) == 0) {
      HOLOSCAN_LOG_WARN(
          "sigmf_file_sink: OVERFLOW (writer behind real time) -- dropped frame {}: {} signal(s), "
          "{} IQ sample(s) ({} original-rate sample(s)). Cumulative dropped: {} signal(s), "
          "{} IQ sample(s), {} original-rate sample(s) across {} frame(s). "
          "Raise max_queued_batches or emit_stride, or use faster storage.",
          frame_number, batch_signals, batch_samples, batch_orig_samples,
          snippets_dropped_, samples_dropped_, orig_samples_dropped_, batches_dropped_);
    }
    return;
  }

  // Stage device->host NOW (fast) so the pooled device IQ frees as `batch` goes out of scope; the
  // async writer thread only touches host memory -> disk.
  std::vector<snip::HostSnippet> host_batch;
  host_batch.reserve(batch.snippets.size());
  for (const auto& snippet : batch.snippets) {
    host_batch.push_back(stage_to_host(snippet));
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(std::move(host_batch));
  }
  cv_.notify_one();
}

void SigmfFileSinkOp::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  if (writer_thread_.joinable()) {
    writer_thread_.join();
  }
  if (batches_dropped_ > 0) {
    HOLOSCAN_LOG_WARN(
        "sigmf_file_sink: OVERFLOW SUMMARY -- {} file(s) written; DROPPED {} frame(s): {} signal(s), "
        "{} IQ sample(s), {} original-rate sample(s) were not saved (writer could not keep up with "
        "real time).",
        files_written_, batches_dropped_, snippets_dropped_, samples_dropped_, orig_samples_dropped_);
  } else {
    HOLOSCAN_LOG_INFO("sigmf_file_sink: total files written {}, no overflow (0 dropped).",
                      files_written_);
  }
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

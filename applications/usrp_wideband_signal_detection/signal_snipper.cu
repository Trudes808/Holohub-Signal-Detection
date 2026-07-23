// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "signal_snipper.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace holoscan::ops {

void SignalSnipperOp::setup(holoscan::OperatorSpec& spec) {
  // The IQ tap (from CHDR converter / offline source) DRIVES compute: it arrives one frame at a
  // time in order and leads the mask by the detector-pipeline latency. Buffering per IQ frame keeps
  // the ring current, so a lagging mask always finds its (older) frame still buffered.
  auto& iq_port = spec.input<iq_in_t>("iq_in", holoscan::IOSpec::IOSize{16});
  iq_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));

  // Masks lag the IQ; drain whatever is available each compute (optional so it never stalls). The
  // eval's drain frames keep compute firing after the last real IQ so trailing masks get processed.
  auto& mask_port = spec.input<DetectorMaskMessage>("mask_in", holoscan::IOSpec::IOSize{16});
  mask_port.condition(holoscan::ConditionType::kNone);

  // Default (DownstreamMessageAffordable) condition + a deep queue: under live load the file sink
  // can lag, so the snipper must throttle rather than overflow the transmitter (kNone would remove
  // backpressure and make emit throw GXF_EXCEEDING_PREALLOCATED_SIZE fatally).
  spec.output<SnippetBatchMessage>("snippets_out", holoscan::IOSpec::IOSize{16});

  spec.param(mode_, "mode", "Mode", "Snip mode: 'time_only' or 'frequency'.", std::string("time_only"));
  spec.param(oversample_percent_, "oversample_percent", "Oversample percent",
             "Extra bandwidth retained beyond the detected bandwidth when downsampling.", 25.0);
  spec.param(enable_downsample_, "enable_downsample", "Enable downsample",
             "In frequency mode, decimate each signal to the minimum rate (plus oversample).", true);
  spec.param(bandwidth_margin_hz_, "bandwidth_margin_hz", "Bandwidth margin Hz",
             "Absolute Hz added to the detected bandwidth before low-pass filtering.", 0.0);
  spec.param(min_box_pixels_, "min_box_pixels", "Min box pixels",
             "Discard connected components smaller than this many mask pixels (speckle filter).", 256);
  spec.param(merge_gap_rows_, "merge_gap_rows", "Merge gap rows",
             "Coalesce component boxes within this many mask rows (time) of each other.", 16);
  spec.param(merge_gap_cols_, "merge_gap_cols", "Merge gap cols",
             "Coalesce component boxes within this many mask cols (frequency) of each other.", 80);
  spec.param(fir_num_taps_, "fir_num_taps", "FIR taps", "Low-pass FIR length (forced odd).", 129);
  spec.param(ring_depth_, "ring_depth", "Ring depth",
             "Max full-rate IQ frames buffered while awaiting their mask (memory cap; must exceed "
             "the detector-pipeline latency in frames -- offline greedy scheduling can run the "
             "source ~15-20 frames ahead of the detector).", 32);
  spec.param(channel_filter_, "channel_filter", "Channel filter",
             "Only process this channel (-1 = all).", -1);
  spec.param(center_frequency_hz_, "center_frequency_hz", "Center frequency Hz",
             "RF center used if not present in metadata.", 0.0);
  spec.param(sample_rate_hz_, "sample_rate_hz", "Sample rate Hz",
             "Full stream sample rate used if not present in metadata.", 0.0);
}

void SignalSnipperOp::initialize() {
  holoscan::Operator::initialize();
  // Draining multiple IQ messages plus a mask per compute merges metadata from several messages;
  // last-writer-wins avoids merge-conflict throws.
  metadata_policy(holoscan::MetadataPolicy::kUpdate);

  pool_ = std::make_shared<DeviceBufferPool>();
  if (cudaStreamCreateWithFlags(&snip_stream_, cudaStreamNonBlocking) != cudaSuccess) {
    throw std::runtime_error("SignalSnipperOp: failed to create CUDA stream");
  }
  const int depth = std::max(2, ring_depth_.get()) + 2;
  event_pool_.resize(static_cast<size_t>(depth));
  for (auto& ev : event_pool_) {
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  }
}

void SignalSnipperOp::ingest_iq(holoscan::InputContext& op_input) {
  while (true) {
    auto in = op_input.receive<iq_in_t>("iq_in");
    if (!in) {
      break;
    }
    auto tensor = std::get<0>(in.value());
    cudaStream_t producer_stream = std::get<1>(in.value());
    const uint64_t n = static_cast<uint64_t>(tensor.Size(0)) * static_cast<uint64_t>(tensor.Size(1));
    if (n == 0) {
      ++iq_arrival_counter_;
      continue;
    }

    ++iq_arrival_counter_;  // matches the FFT's per-input counter == mask.frame_number
    const uint64_t frame_number = iq_arrival_counter_;

    RingEntry entry;
    entry.frame_number = frame_number;
    entry.n_iq = n;
    entry.device_iq = pool_->acquire(n);

    // Order the copy after the producer finished, then run it on our own stream so all ring copies
    // and downstream DDC are serialized on snip_stream_.
    cudaEvent_t ev = event_pool_[frame_number % event_pool_.size()];
    cudaEventRecord(ev, producer_stream);
    cudaStreamWaitEvent(snip_stream_, ev, 0);
    cudaMemcpyAsync(entry.device_iq.get(),
                    tensor.Data(),
                    n * sizeof(SnipComplex),
                    cudaMemcpyDeviceToDevice,
                    snip_stream_);

    ring_.push_back(std::move(entry));
    prune_ring();
  }
}

void SignalSnipperOp::prune_ring() {
  // Drop frames already covered by a processed mask (masks arrive in increasing frame order).
  while (!ring_.empty() && ring_.front().frame_number <= last_processed_mask_frame_) {
    ring_.pop_front();
  }
  // Memory backstop: if masks stop arriving, cap the buffer and drop the oldest.
  const size_t cap = static_cast<size_t>(std::max(2, ring_depth_.get()));
  while (ring_.size() > cap) {
    HOLOSCAN_LOG_WARN("signal_snipper: IQ ring exceeded cap {} (mask lag too large?); dropping frame {}.",
                      cap, ring_.front().frame_number);
    ring_.pop_front();
  }
}

SignalSnipperOp::RingEntry* SignalSnipperOp::find_ring_entry(uint64_t frame_number) {
  for (auto& entry : ring_) {
    if (entry.frame_number == frame_number) {
      return &entry;
    }
  }
  return nullptr;
}

namespace {

struct TimeInterval {
  uint64_t start = 0;  // local sample offset within the frame
  uint64_t end = 0;
  std::vector<snip::PhysicalRegion> regions;  // signals overlapping this interval
};

// Merge component time ranges (local sample offsets) into disjoint intervals, tracking which regions
// fall in each so time-only snippets can annotate every signal they contain.
std::vector<TimeInterval> merge_time_intervals(const std::vector<snip::PhysicalRegion>& regions) {
  std::vector<snip::PhysicalRegion> sorted = regions;
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    return a.local_start < b.local_start;
  });
  std::vector<TimeInterval> intervals;
  for (const auto& region : sorted) {
    const uint64_t start = region.local_start;
    const uint64_t end = region.local_start + region.sample_count;
    if (!intervals.empty() && start <= intervals.back().end) {
      intervals.back().end = std::max(intervals.back().end, end);
      intervals.back().regions.push_back(region);
    } else {
      TimeInterval interval;
      interval.start = start;
      interval.end = end;
      interval.regions.push_back(region);
      intervals.push_back(std::move(interval));
    }
  }
  return intervals;
}

}  // namespace

void SignalSnipperOp::compute(holoscan::InputContext& op_input,
                              holoscan::OutputContext& op_output,
                              holoscan::ExecutionContext&) {
  // IQ drives compute (leads the mask); buffer all available frames, then drain any masks whose IQ
  // is now buffered.
  ingest_iq(op_input);

  // Drain every available mask into ONE batch, then emit exactly once (see process_mask note).
  SnippetBatchMessage batch;
  while (true) {
    auto mask_in = op_input.receive<DetectorMaskMessage>("mask_in");
    if (!mask_in) {
      break;
    }
    process_mask(mask_in.value(), batch);
  }

  if (!batch.snippets.empty()) {
    // Ensure all copies / DDC on snip_stream_ completed before the batch (and its device buffers)
    // travel downstream to the file sink, which reads them on its own stream.
    cudaStreamSynchronize(snip_stream_);
    snippets_emitted_ += batch.snippets.size();
    op_output.emit(std::move(batch), "snippets_out");
  }
}

void SignalSnipperOp::process_mask(const DetectorMaskMessage& mask, SnippetBatchMessage& batch) {
  const int configured_channel = channel_filter_.get();
  if (configured_channel >= 0 && mask.channel >= 0 && mask.channel != configured_channel) {
    return;
  }

  // Masks arrive in increasing frame order; record progress so older IQ frames can be pruned.
  if (mask.frame_number > last_processed_mask_frame_) {
    last_processed_mask_frame_ = mask.frame_number;
  }

  if (mask.width <= 0 || mask.height <= 0) {
    prune_ring();
    return;
  }

  RingEntry* entry = find_ring_entry(mask.frame_number);
  if (entry == nullptr) {
    HOLOSCAN_LOG_WARN("signal_snipper: no buffered IQ for mask frame {} (ring size {}, cap {}); dropping.",
                      mask.frame_number, ring_.size(), ring_depth_.get());
    prune_ring();
    return;
  }

  // Resolve the TRUE delivered sample rate, preferring pipeline metadata so it flows automatically:
  //   1. rx_sample_rate_hz  -- stamped by the CHDR converter when chdr_converter.channel_sample_rates_hz
  //      is set (live/loopback); this is the actual radio rate.
  //   2. configured sample_rate_hz (>0) -- explicit override, and the value the offline driver injects
  //      from the SigMF core:sample_rate.
  //   3. sample_rate_hz / span metadata -- the FFT's derived span (already the SigMF rate offline, or
  //      the channel_sample_rates_hz-derived rate live). Only nominal (e.g. 500e6) if none of the
  //      above provided a true rate.
  auto meta = metadata();
  double sample_rate = 0.0;
  if (meta && meta->has_key("rx_sample_rate_hz")) {
    sample_rate = meta->get<double>("rx_sample_rate_hz", 0.0);
  }
  if (sample_rate <= 0.0) {
    sample_rate = sample_rate_hz_.get();
  }
  if (sample_rate <= 0.0 && meta) {
    if (meta->has_key("sample_rate_hz")) {
      sample_rate = meta->get<double>("sample_rate_hz", 0.0);
    } else if (meta->has_key("span")) {
      sample_rate = static_cast<double>(meta->get<uint64_t>("span", 0));
    }
  }
  // Center frequency: prefer the CHDR-stamped RF center, else the configured fallback.
  double center_freq = center_frequency_hz_.get();
  if (meta && meta->has_key("rx_center_frequency_hz")) {
    center_freq = meta->get<double>("rx_center_frequency_hz", center_freq);
  }

  snip::FrameGeometry geom;
  geom.mask_rows = mask.height;
  geom.mask_cols = mask.width;
  geom.frame_sample_count = entry->n_iq;
  geom.sample_rate_hz = sample_rate;
  geom.center_freq_hz = center_freq;
  geom.frame_sample_start =
      mask.file_offset_complex > 0 ? mask.file_offset_complex
                                   : (mask.frame_number > 0 ? (mask.frame_number - 1) * entry->n_iq : 0);

  // Copy the small mask to the host (reused scratch) and cluster it into per-signal boxes.
  const size_t mask_bytes = static_cast<size_t>(mask.width) * static_cast<size_t>(mask.height);
  host_mask_.resize(mask_bytes);
  if (mask.device_pixels) {
    cudaMemcpy(host_mask_.data(), mask.device_pixels.get(), mask_bytes, cudaMemcpyDeviceToHost);
  } else if (mask.pixels.size() >= mask_bytes) {
    std::copy_n(mask.pixels.begin(), mask_bytes, host_mask_.begin());
  } else {
    return;
  }

  auto boxes =
      snip::label_components(host_mask_, mask.height, mask.width, min_box_pixels_.get(), cc_scratch_);
  boxes = snip::merge_boxes(std::move(boxes), merge_gap_rows_.get(), merge_gap_cols_.get());
  if (boxes.empty()) {
    return;
  }

  std::vector<snip::PhysicalRegion> regions;
  regions.reserve(boxes.size());
  for (const auto& box : boxes) {
    regions.push_back(snip::map_box_to_physical(box, geom));
  }

  snip::SnipDspParams dsp;
  dsp.oversample_percent = oversample_percent_.get();
  dsp.enable_downsample = enable_downsample_.get();
  dsp.bandwidth_margin_hz = bandwidth_margin_hz_.get();
  dsp.fir_num_taps = fir_num_taps_.get();

  const bool frequency_mode = mode_.get() == "frequency";
  const SnipComplex* frame_iq = entry->device_iq.get();
  const uint64_t frame_n = entry->n_iq;

  // Stamp the batch with the most recent frame/channel processed this tick (each snippet also
  // carries its own frame_number for downstream reassembly).
  batch.frame_number = mask.frame_number;
  batch.channel = mask.channel;

  auto make_annotation = [](const snip::PhysicalRegion& region) {
    SnipAnnotation ann;
    ann.freq_lower_hz = region.freq_lower_hz;
    ann.freq_upper_hz = region.freq_upper_hz;
    ann.label = "waveform_detection";
    ann.kind = "waveform";
    return ann;
  };

  if (frequency_mode) {
    for (const auto& region : regions) {
      snip::SnippetIq iq = snip::ddc_extract(frame_iq, frame_n, region, dsp, *pool_, snip_stream_);
      if (!iq.device_iq || iq.n_iq == 0) {
        continue;
      }
      SignalSnippet snippet;
      snippet.frame_number = mask.frame_number;
      snippet.channel = mask.channel;
      snippet.orig_sample_start = region.sample_start;
      snippet.orig_sample_count = region.sample_count;
      snippet.orig_sample_rate_hz = geom.sample_rate_hz;
      snippet.sample_rate_hz = iq.sample_rate_hz;
      snippet.center_freq_hz = region.freq_center_hz;
      snippet.n_iq = iq.n_iq;
      snippet.device_iq = std::move(iq.device_iq);
      snippet.annotations.push_back(make_annotation(region));
      batch.snippets.push_back(std::move(snippet));
    }
  } else {
    // time_only: one full-band snippet per merged time interval; annotate every signal within it.
    for (const auto& interval : merge_time_intervals(regions)) {
      const uint64_t count = interval.end - interval.start;
      snip::SnippetIq iq = snip::copy_time_slice(frame_iq, frame_n, interval.start, count,
                                                 geom.sample_rate_hz, *pool_, snip_stream_);
      if (!iq.device_iq || iq.n_iq == 0) {
        continue;
      }
      SignalSnippet snippet;
      snippet.frame_number = mask.frame_number;
      snippet.channel = mask.channel;
      snippet.orig_sample_start = geom.frame_sample_start + interval.start;
      snippet.orig_sample_count = count;
      snippet.orig_sample_rate_hz = geom.sample_rate_hz;
      snippet.sample_rate_hz = geom.sample_rate_hz;
      snippet.center_freq_hz = geom.center_freq_hz;
      snippet.n_iq = iq.n_iq;
      snippet.device_iq = std::move(iq.device_iq);
      for (const auto& region : interval.regions) {
        snippet.annotations.push_back(make_annotation(region));
      }
      batch.snippets.push_back(std::move(snippet));
    }
  }

  ++masks_processed_;
  // This frame's mask is done; drop it (and any older) from the ring. (The batch is emitted once,
  // after all masks in this compute tick are processed -- see compute().)
  prune_ring();
}

void SignalSnipperOp::stop() {
  if (snip_stream_ != nullptr) {
    cudaStreamSynchronize(snip_stream_);
    cudaStreamDestroy(snip_stream_);
    snip_stream_ = nullptr;
  }
  for (auto& ev : event_pool_) {
    if (ev != nullptr) {
      cudaEventDestroy(ev);
    }
  }
  event_pool_.clear();
  ring_.clear();
  HOLOSCAN_LOG_INFO("signal_snipper: processed {} masks, emitted {} snippets.",
                    masks_processed_, snippets_emitted_);
  holoscan::Operator::stop();
}

}  // namespace holoscan::ops

/*
 * SPDX-FileCopyrightText: 2026 National Instruments Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "chdr_rx.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <sstream>
#include <stdexcept>

using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

using namespace std::complex_literals;

// CUDA kernel to process an individual CHDR packet
__global__ void place_packet_data_kernel(complex* out,
                                         const void* const* const __restrict__ in,
                                         const int cur_idx,
                                         const int num_complex_samples_per_packet,
                                         const int packets_in_batch
  ) {
  // Warmup
  if (out == nullptr)
    return;

  const int packet_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (packet_index >= packets_in_batch) {
    return;
  }

  // The in pointer is an array holding a pointer to the samples for
  // an entire batch (in[640]).
  // blockIdx.x is the packet row and threadIdx.x the packet index
  // This assumes interleaved 16-bit short IQ samples
  const int16_t *samples = reinterpret_cast<const int16_t*>(
          in[(blockIdx.x * blockDim.x) + threadIdx.x]);

  // Scale the int16 values to -1.0 thru +1.0 by dividing by 2^15 - 1 (0x7FFF)
  constexpr float scalar = 1.0 / 0x7FFF;

  // The out pointer is a 4d tensor with structure:
  // 1                        2
  // ---------------------------------------------
  // [P1][P2][P3]...[P20]     [P1][P2][P3]...[P20]
  // [P21][P22]...[P40]       [P21][P22]...[P40]
  // ...                      ...
  // [P12780]...[P12800]      [P12780]...[P12800]
  // We want to get to the index of one of these packets.
  // gridDim.x is num_ffts_per_batch
  // blockDim.x is num_packets_per_fft
  // blockIdx.x is the packet row
  // threadIdx.x is the packet index
  // First, get to the right section of the output tensor (1 or 2),
  // then, index into the row,
  // then, index into the packet
  size_t offset = (num_complex_samples_per_packet * blockDim.x * gridDim.x * cur_idx)
                + (num_complex_samples_per_packet * blockDim.x * blockIdx.x)
                + (num_complex_samples_per_packet * threadIdx.x);

  // Copy data while performing an endian flip and casting to complex float
  for (size_t i = 0; i < num_complex_samples_per_packet; ++i) {
    // Casting includes conversion from network order on little-endian systems
    out[offset + i] = complex(static_cast<float>(samples[i * 2]) * scalar,
                      static_cast<float>(samples[(i * 2) + 1]) * scalar);
  }
}

void place_packet_data(complex* out,
                       const void* const* const in,
                       const uint16_t cur_idx,
                       const int num_ffts_per_batch,
                       const int num_packets_per_fft,
                       const int num_complex_samples_per_packet,
                       cudaStream_t stream,
                       const int packets_in_batch) {
  // CUDA execution config <<<Dg, Db, Ns, S>>> where:
  // Dg: dimensionality of the grid of blocks
  // Db: dimensionality of the block of threads
  // Ns: number of bytes in shared memory that is dynamically
  //     allocated _per block_ for this call in addition to
  //     the statically allocated memory
  //  S: associated CUDA stream
  // At this point, we're processing num_ffts_per_batch * num_packets_per_fft packets
  // (e.g. 125 * 20 = 2,500).
  // So, let's launch a grid for every num_packets_per_fft and a thread for every packet.
  // This would make blockIdx.x the packet row and threadIdx.x the packet.
  place_packet_data_kernel<<<
      num_ffts_per_batch,
      num_packets_per_fft,
      num_packets_per_fft * sizeof(int), stream>>>(
          out,
          in,
          cur_idx,
          num_complex_samples_per_packet,
          packets_in_batch);
}

namespace holoscan::ops {

namespace {

constexpr uint64_t kChdrSummaryPeriodNs = 1000000000ULL;

uint64_t steady_time_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

double elapsed_ms_since(uint64_t start_ns) {
  if (start_ns == 0) {
    return 0.0;
  }
  const uint64_t now_ns = steady_time_ns();
  if (now_ns <= start_ns) {
    return 0.0;
  }
  return static_cast<double>(now_ns - start_ns) / 1.0e6;
}

std::string trim_copy(const std::string& value) {
  const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch);
  });
  const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
    return std::isspace(ch);
  }).base();
  if (begin >= end) {
    return {};
  }
  return std::string(begin, end);
}

std::vector<std::string> split_csv_strings(const std::string& values) {
  std::vector<std::string> parsed;
  std::stringstream stream(values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const auto trimmed_token = trim_copy(token);
    if (!trimmed_token.empty()) {
      parsed.push_back(trimmed_token);
    }
  }
  return parsed;
}

void maybe_log_channel_summary(const std::shared_ptr<ChdrConverterOpRx::Channel>& channel) {
  if (channel->periodic_summary_start_ns == 0 ||
      channel->periodic_summary_last_ns <= channel->periodic_summary_start_ns) {
    return;
  }

  const uint64_t window_ns = channel->periodic_summary_last_ns - channel->periodic_summary_start_ns;
  if (window_ns < kChdrSummaryPeriodNs) {
    return;
  }

  const double window_s = static_cast<double>(window_ns) / 1.0e9;
  const double avg_mpps =
      (window_s > 0.0) ? (static_cast<double>(channel->periodic_summary_packets) / window_s / 1.0e6)
                       : 0.0;
  HOLOSCAN_LOG_INFO(
      "CHDR summary ch={} window_s={:.3f} bursts={} packets={} rate={:.3f} Mpps queued={} emitted={} empty_polls={} backlog_events={} out_q_depth={} max_out_q_depth={} aggr_pkts_recv={} max_burst_packets={} refcounted_bursts={} partial_drops={}",
      channel->channel_num,
      window_s,
      channel->periodic_summary_bursts,
      channel->periodic_summary_packets,
      avg_mpps,
      channel->periodic_summary_batches_queued,
      channel->periodic_summary_batches_emitted,
      channel->periodic_summary_empty_polls,
      channel->periodic_summary_backlog_events,
      channel->out_q.size(),
      channel->max_out_q_depth,
      channel->aggr_pkts_recv,
      channel->max_burst_packets,
      channel->burst_refcounts.size(),
      channel->timeout_like_partial_drains);

  channel->periodic_summary_start_ns = channel->periodic_summary_last_ns;
  channel->periodic_summary_batches_queued = 0;
  channel->periodic_summary_batches_emitted = 0;
  channel->periodic_summary_packets = 0;
  channel->periodic_summary_bursts = 0;
  channel->periodic_summary_empty_polls = 0;
  channel->periodic_summary_backlog_events = 0;
}

uint16_t resolve_channel_from_burst_flow(BurstParams* burst,
                                         uint16_t fallback_channel,
                                         uint16_t num_channels) {
  if (burst == nullptr || num_channels == 0) {
    return fallback_channel;
  }

  const int burst_packets = get_num_packets(burst);
  if (burst_packets <= 0) {
    return fallback_channel;
  }

  const uint16_t first_flow_id = get_packet_flow_id(burst, 0);
  bool mixed_flow_ids = false;
  for (int packet_index = 1; packet_index < burst_packets; ++packet_index) {
    if (get_packet_flow_id(burst, packet_index) != first_flow_id) {
      mixed_flow_ids = true;
      break;
    }
  }

  if (mixed_flow_ids) {
    HOLOSCAN_LOG_WARN(
        "CHDR burst on queue {} contains mixed flow IDs; falling back to queue-derived channel assignment",
        fallback_channel);
    return fallback_channel;
  }

  if (first_flow_id >= num_channels) {
    HOLOSCAN_LOG_WARN(
        "CHDR burst on queue {} matched flow_id={} outside configured channel range {}; falling back to queue-derived channel assignment",
        fallback_channel,
        first_flow_id,
        num_channels);
    return fallback_channel;
  }

  if (first_flow_id != fallback_channel) {
    HOLOSCAN_LOG_WARN(
        "CHDR queue/flow mismatch: queue {} received flow_id {}. Routing burst to logical channel {} instead of queue index.",
        fallback_channel,
        first_flow_id,
        first_flow_id);
  }

  return first_flow_id;
}

}  // namespace

int ChdrConverterOpRx::max_inflight_batches() const {
  return std::max(1, std::min<int>(num_concurrent, num_simul_batches_.get()));
}

void ChdrConverterOpRx::setup(OperatorSpec& spec) {
  // Use one output port per channel so we can drain more than one completed
  // batch per compute() call without publishing multiple messages on the same
  // typed output port in a single tick.
  spec.output<out_t>("out0", holoscan::IOSpec::IOSize{16});
  spec.output<out_t>("out1", holoscan::IOSpec::IOSize{16});

  // Data tensor configuration
  // Each packet contains 1024 samples
  // We want to batch up 20 packets for an FFT calculation, so
  // we want 20x1024.
  // We want to perform 625 FFT calculations at once, so we want
  // to batch up 125x20x1024 samples.
  // We want to have 2 data buffers ping-ponging between processing
  // and accumulation, so we want to hold 2x125x20x1024 samples.
  // rf_data is a tensor_t which represents this data. The last
  // dimension (20x1024 in this example) is collapsed into one as
  // downstream operators don't care about how many packets were
  // accumulated.
  spec.param<uint16_t>(num_complex_samples_per_packet_,
      "num_complex_samples_per_packet",
      "Number of complex samples per packet",
      "Number of complex samples per CHDR packet", 1024);
  spec.param<uint16_t>(num_packets_per_fft_,
      "num_packets_per_fft",
      "Number of packets per FFT",
      "Number of packets per individual FFT computation", 20);
  spec.param<uint16_t>(num_ffts_per_batch_,
      "num_ffts_per_batch",
      "Number of ffts per batch",
      "Number of fft data batches batches to send for processing at once", 125);
  spec.param<uint16_t>(num_simul_batches_,
      "num_simul_batches",
      "Number of simultaneous batches",
      "Number of simultaneous batches to accumulate/process at once", 2);
  spec.param<uint16_t>(num_channels_,
      "num_channels",
      "Number of channels",
      "Number of channels to process", 2);
  spec.param<std::string>(interface_name_,
      "interface_name",
      "Name of the RX port",
      "Name of the RX port from the advanced_network config",
      "sdr_data");
  spec.param<std::string>(interface_names_,
      "interface_names",
      "Names of RX ports",
      "Comma-separated names of RX ports from the advanced_network config. When set, channels are assigned to interfaces and queues in listed order.",
      "");
  spec.param<bool>(log_data_,
      "log_data",
      "Log Data",
      "If true, log detailed data information for debugging.", false);
  spec.param<bool>(log_packets_,
      "log_packets",
      "Log Packets",
      "If true, log detailed packet information for debugging.", false);
  spec.param<uint32_t>(partial_batch_drop_timeout_ms_,
      "partial_batch_drop_timeout_ms",
      "Partial Batch Drop Timeout Ms",
      "Drop an incomplete aggregated batch after this many milliseconds without any new packets on that channel.",
      250);
  spec.param<std::string>(channel_center_frequencies_hz_,
      "channel_center_frequencies_hz",
      "Channel Center Frequencies Hz",
      "Comma-separated per-channel tuned RX center frequencies in Hz for downstream metadata.",
      "");
  spec.param<std::string>(channel_sample_rates_hz_,
      "channel_sample_rates_hz",
      "Channel Sample Rates Hz",
      "Comma-separated per-channel RX sample rates in Hz for downstream metadata.",
      "");
}

std::vector<std::optional<double>> ChdrConverterOpRx::parse_channel_values(
        const std::string& values,
        const char* field_name) const {
  std::vector<std::optional<double>> parsed(num_channels_.get(), std::nullopt);
  const auto trimmed_values = trim_copy(values);
  if (trimmed_values.empty()) {
    return parsed;
  }

  std::vector<double> numeric_values;
  std::stringstream stream(trimmed_values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const auto trimmed_token = trim_copy(token);
    if (trimmed_token.empty()) {
      continue;
    }
    try {
      numeric_values.push_back(std::stod(trimmed_token));
    } catch (const std::exception&) {
      HOLOSCAN_LOG_ERROR("Invalid {} value '{}'", field_name, trimmed_token);
      exit(1);
    }
  }

  if (numeric_values.empty()) {
    return parsed;
  }

  if (numeric_values.size() == 1) {
    std::fill(parsed.begin(), parsed.end(), numeric_values.front());
    return parsed;
  }

  if (numeric_values.size() != num_channels_.get()) {
    HOLOSCAN_LOG_ERROR(
        "Configured {} count {} must be 1 or match num_channels {}",
        field_name,
        numeric_values.size(),
        num_channels_.get());
    exit(1);
  }

  for (size_t index = 0; index < numeric_values.size(); ++index) {
    parsed[index] = numeric_values[index];
  }
  return parsed;
}

std::vector<std::string> ChdrConverterOpRx::parse_interface_names() const {
  auto interface_names = split_csv_strings(interface_names_.get());
  if (!interface_names.empty()) {
    return interface_names;
  }

  const auto interface_name = trim_copy(interface_name_.get());
  if (!interface_name.empty()) {
    return {interface_name};
  }

  HOLOSCAN_LOG_ERROR("CHDR converter requires at least one configured RX interface name");
  exit(1);
}

void ChdrConverterOpRx::initialize() {
  holoscan::Operator::initialize();

  if (num_simul_batches_.get() > num_concurrent) {
    HOLOSCAN_LOG_ERROR("Configured num_simul_batches={} exceeds supported in-flight batch slots {}",
                       num_simul_batches_.get(),
                       num_concurrent);
    exit(1);
  }

  if (num_channels_.get() > 2) {
    HOLOSCAN_LOG_ERROR("ChdrConverterOpRx currently supports at most 2 output channels, got {}",
                       num_channels_.get());
    exit(1);
  }

  const auto interface_names = parse_interface_names();
  use_single_channel_fast_path_ = (interface_names.size() == 1 && num_channels_.get() == 1);
  use_flow_id_routing_ = (interface_names.size() == 1 && num_channels_.get() > 1);

  struct PortInfo {
    std::string interface_name;
    int port_id = -1;
    uint16_t num_rx_queues = 0;
  };

  std::vector<PortInfo> port_infos;
  port_infos.reserve(interface_names.size());
  for (const auto& interface_name : interface_names) {
    const int port_id = get_port_id(interface_name);
    if (port_id == -1) {
      HOLOSCAN_LOG_ERROR("Invalid RX port {} specified in the config", interface_name);
      exit(1);
    }

    const auto num_rx_queues = get_num_rx_queues(port_id);
    if (num_rx_queues <= 0) {
      HOLOSCAN_LOG_ERROR("RX port {} has no configured RX queues", interface_name);
      exit(1);
    }

    PortInfo port_info;
    port_info.interface_name = interface_name;
    port_info.port_id = port_id;
    port_info.num_rx_queues = static_cast<uint16_t>(num_rx_queues);
    port_infos.push_back(port_info);
  }

  rx_sources_.clear();
  for (uint16_t queue_id = 0; rx_sources_.size() < num_channels_.get(); ++queue_id) {
    bool added_any_source = false;
    for (const auto& port_info : port_infos) {
      if (queue_id >= port_info.num_rx_queues || rx_sources_.size() >= num_channels_.get()) {
        continue;
      }

      RxSource rx_source;
      rx_source.port_id = port_info.port_id;
      rx_source.queue_id = queue_id;
      rx_source.fallback_channel = static_cast<uint16_t>(rx_sources_.size());
      rx_source.interface_name = port_info.interface_name;
      rx_sources_.push_back(rx_source);
      added_any_source = true;
    }

    if (!added_any_source) {
      break;
    }
  }

  if (rx_sources_.size() < num_channels_.get()) {
    HOLOSCAN_LOG_ERROR(
        "Configured RX interfaces expose {} queue(s), but num_channels={} requires at least one queue per channel",
        rx_sources_.size(),
        num_channels_.get());
    exit(1);
  }

  if (use_single_channel_fast_path_) {
    single_channel_port_id_ = rx_sources_.front().port_id;
    single_channel_queue_id_ = rx_sources_.front().queue_id;
  }

  HOLOSCAN_LOG_INFO(
      "CHDR RX source mapping mode={} sources={} num_channels={}",
      use_single_channel_fast_path_
          ? "single-interface-single-channel-fast-path"
          : (use_flow_id_routing_ ? "single-interface-flow-routing" : "multi-interface-fixed-routing"),
      rx_sources_.size(),
      num_channels_.get());
  for (const auto& rx_source : rx_sources_) {
    HOLOSCAN_LOG_INFO(
        "CHDR RX source mapped interface={} port_id={} queue_id={} -> logical_channel={}",
        rx_source.interface_name,
        rx_source.port_id,
        rx_source.queue_id,
        rx_source.fallback_channel);
  }

  num_packets_per_batch = num_ffts_per_batch_.get() * num_packets_per_fft_.get();
  center_frequency_by_channel_ = parse_channel_values(channel_center_frequencies_hz_.get(),
                                                      "channel_center_frequencies_hz");
  sample_rate_by_channel_ = parse_channel_values(channel_sample_rates_hz_.get(),
                                                 "channel_sample_rates_hz");

  for (uint16_t channel_num = 0; channel_num < num_channels_.get(); channel_num++) {
    auto new_channel = std::make_shared<struct Channel>();
    new_channel->channel_num = channel_num;
    make_tensor(new_channel->rf_data,
                {num_simul_batches_.get(),
                 num_ffts_per_batch_.get(),
                 num_packets_per_fft_.get() * num_complex_samples_per_packet_.get()});

    // Allocate memory and create CUDA streams for each concurrent batch
    for (int n = 0; n < num_simul_batches_.get(); n++) {
      cudaMallocHost((void**)&new_channel->h_dev_ptrs[n], sizeof(void*) * num_packets_per_batch);

      cudaStreamCreateWithFlags(&new_channel->streams[n], cudaStreamNonBlocking);
      cudaEventCreate(&new_channel->events[n]);
      // Warmup
      place_packet_data(nullptr,
                        nullptr,
                        0,
                        num_ffts_per_batch_.get(),
                        num_packets_per_fft_.get(),
                        num_complex_samples_per_packet_.get(),
                        new_channel->streams[n],
                        num_packets_per_batch);
      cudaStreamSynchronize(new_channel->streams[n]);
    }

    channel_list.push_back(new_channel);
  }
}

std::optional<ChdrConverterOpRx::RxMsg> ChdrConverterOpRx::free_buf(
        std::shared_ptr<struct Channel> channel) {
  if (!channel->out_q.empty()) {
    auto first = channel->out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      const double release_latency_ms = elapsed_ms_since(first.queued_ns);
      channel->release_samples++;
      channel->total_release_latency_ms += release_latency_ms;
      channel->max_release_latency_ms = std::max(channel->max_release_latency_ms, release_latency_ms);
      for (auto m = 0; m < first.num_batches; m++) {
        release_burst_ref(channel, first.msg[m]);
      }
      channel->out_q.pop();
      return std::optional<ChdrConverterOpRx::RxMsg>{first};
    }
  }
  return std::nullopt;
}

void ChdrConverterOpRx::retain_burst_ref(
        std::shared_ptr<struct Channel> channel,
        BurstParams* burst) {
  channel->burst_refcounts[burst]++;
}

void ChdrConverterOpRx::release_burst_ref(
        std::shared_ptr<struct Channel> channel,
        BurstParams* burst) {
  auto ref = channel->burst_refcounts.find(burst);
  if (ref == channel->burst_refcounts.end()) {
    HOLOSCAN_LOG_ERROR("Missing burst refcount while releasing CHDR burst on channel {}",
                       channel->channel_num);
    free_all_packets_and_burst_rx(burst);
    return;
  }

  if (--ref->second == 0) {
    channel->burst_refcounts.erase(ref);
    free_all_packets_and_burst_rx(burst);
  }
}

void ChdrConverterOpRx::flush_partial_batch(
        std::shared_ptr<struct Channel> channel,
        const char* reason) {
  if (!channel || channel->aggr_pkts_recv == 0 || channel->cur_msg.num_batches == 0) {
    return;
  }

  const uint64_t held_packets = channel->aggr_pkts_recv;
  const int held_bursts = channel->cur_msg.num_batches;
  channel->timeout_like_partial_drains++;
  HOLOSCAN_LOG_WARN(
      "Flushing partial CHDR batch on channel {} reason={} held_packets={} held_bursts={} refcounted_bursts_before_flush={}",
      channel->channel_num,
      reason,
      held_packets,
      held_bursts,
      channel->burst_refcounts.size());
  queue_completed_batch(channel, static_cast<uint32_t>(held_packets), true);
}

void ChdrConverterOpRx::queue_completed_batch(
        std::shared_ptr<struct Channel> channel,
        uint32_t packets_in_batch,
        bool partial_batch) {
  const int completed_batch_idx = channel->cur_idx;
  const uint32_t safe_packets_in_batch = std::min<uint32_t>(packets_in_batch, num_packets_per_batch);

  HOLOSCAN_LOG_DEBUG("Aggregated {} packets on channel {} index {} - sending downstream",
                     safe_packets_in_batch,
                     channel->channel_num,
                     completed_batch_idx);

  if (partial_batch && safe_packets_in_batch < num_packets_per_batch) {
    auto batch = slice<2>(channel->rf_data,
                          {static_cast<index_t>(completed_batch_idx), 0, 0},
                          {matxDropDim, matxEnd, matxEnd});
    cudaMemsetAsync(batch.Data(),
                    0,
                    batch.TotalSize() * sizeof(complex),
                    channel->streams[completed_batch_idx]);
  }

  place_packet_data(channel->rf_data.Data(),
                    channel->h_dev_ptrs[completed_batch_idx],
                    completed_batch_idx,
                    num_ffts_per_batch_.get(),
                    num_packets_per_fft_.get(),
                    num_complex_samples_per_packet_.get(),
                    channel->streams[completed_batch_idx],
                    static_cast<int>(safe_packets_in_batch));

  if (log_data_) {
    HOLOSCAN_LOG_INFO("Inspecting RF channel {} data from thread {} with shape: ({}, {}, {})",
      channel->channel_num, completed_batch_idx,
      channel->rf_data.Size(0), channel->rf_data.Size(1), channel->rf_data.Size(2));
    set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
    print(slice<1>(channel->rf_data, {static_cast<index_t>(completed_batch_idx), 0, 0}, {matxDropDim, matxDropDim, 1024}));
  }

  cudaEventRecord(channel->events[completed_batch_idx], channel->streams[completed_batch_idx]);
  channel->cur_msg.batch_idx = completed_batch_idx;
  channel->cur_msg.stream = channel->streams[completed_batch_idx];
  channel->cur_msg.evt = channel->events[completed_batch_idx];
  channel->cur_msg.queued_ns = steady_time_ns();
  channel->cur_msg.packets_in_batch = safe_packets_in_batch;
  channel->cur_msg.partial_batch = partial_batch;
  channel->out_q.push(channel->cur_msg);
  channel->completed_batches_queued++;
  channel->periodic_summary_batches_queued++;
  channel->max_out_q_depth = std::max(channel->max_out_q_depth, channel->out_q.size());
  channel->cur_msg.num_batches = 0;
  channel->ttl_pkts_recv += safe_packets_in_batch;

  auto ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch", num_ffts_per_batch_.get());
    HOLOSCAN_LOG_ERROR("Error: {}", cudaGetErrorString(ret));
    exit(1);
  }

  channel->aggr_pkts_recv = 0;
  channel->cur_idx = (channel->cur_idx + 1) % num_simul_batches_.get();
}

bool ChdrConverterOpRx::free_bufs_and_emit_arrays(
        OutputContext& op_output,
        std::shared_ptr<struct Channel> channel) {
  const char* output_name = nullptr;
  switch (channel->channel_num) {
    case 0:
      output_name = "out0";
      break;
    case 1:
      output_name = "out1";
      break;
    default:
      HOLOSCAN_LOG_ERROR("Unsupported channel {} in CHDR output drain", channel->channel_num);
      return false;
  }

  std::optional<ChdrConverterOpRx::RxMsg> completed_msg = free_buf(channel);
  if (!completed_msg.has_value()) {
    return false;
  }

  auto meta = metadata();
  meta->clear();
  meta->set("channel_number", channel->channel_num);
  meta->set("chdr_emit_ts_ns", steady_time_ns());
  if (channel->channel_num < center_frequency_by_channel_.size() &&
      center_frequency_by_channel_[channel->channel_num].has_value()) {
    meta->set("rx_center_frequency_hz", center_frequency_by_channel_[channel->channel_num].value());
  }
  if (channel->channel_num < sample_rate_by_channel_.size() &&
      sample_rate_by_channel_[channel->channel_num].has_value()) {
    meta->set("rx_sample_rate_hz", sample_rate_by_channel_[channel->channel_num].value());
  }
  meta->set("chdr_packets_in_batch", completed_msg.value().packets_in_batch);
  meta->set("chdr_expected_packets_in_batch", num_packets_per_batch);
  meta->set("chdr_partial_batch", completed_msg.value().partial_batch);

  auto data = slice<2>(channel->rf_data, {static_cast<index_t>(completed_msg.value().batch_idx), 0, 0},
              {matxDropDim, matxEnd, matxEnd});
  op_output.emit(out_t {data, completed_msg.value().stream}, output_name);
  channel->completed_batches_emitted++;
  channel->periodic_summary_batches_emitted++;
  return true;
}

void ChdrConverterOpRx::release_channel_resources() {
  if (resources_released_) {
    return;
  }
  resources_released_ = true;

  for (auto& channel : channel_list) {
    if (!channel) {
      continue;
    }

    for (int stream_index = 0; stream_index < num_simul_batches_.get(); ++stream_index) {
      if (channel->streams[stream_index] != nullptr) {
        cudaStreamSynchronize(channel->streams[stream_index]);
      }
    }

    while (!channel->out_q.empty()) {
      auto queued_msg = channel->out_q.front();
      channel->out_q.pop();
      for (int batch_index = 0; batch_index < queued_msg.num_batches; ++batch_index) {
        if (queued_msg.msg[batch_index] != nullptr) {
          release_burst_ref(channel, queued_msg.msg[batch_index]);
          queued_msg.msg[batch_index] = nullptr;
        }
      }
    }

    for (int batch_index = 0; batch_index < channel->cur_msg.num_batches; ++batch_index) {
      if (channel->cur_msg.msg[batch_index] != nullptr) {
        release_burst_ref(channel, channel->cur_msg.msg[batch_index]);
        channel->cur_msg.msg[batch_index] = nullptr;
      }
    }
    channel->cur_msg.num_batches = 0;

    for (auto& [burst, refcount] : channel->burst_refcounts) {
      (void)refcount;
      if (burst != nullptr) {
        free_all_packets_and_burst_rx(burst);
      }
    }
    channel->burst_refcounts.clear();

    for (int stream_index = 0; stream_index < num_simul_batches_.get(); ++stream_index) {
      if (channel->events[stream_index] != nullptr) {
        cudaEventDestroy(channel->events[stream_index]);
        channel->events[stream_index] = nullptr;
      }
      if (channel->streams[stream_index] != nullptr) {
        cudaStreamDestroy(channel->streams[stream_index]);
        channel->streams[stream_index] = nullptr;
      }
      if (channel->h_dev_ptrs[stream_index] != nullptr) {
        cudaFreeHost(channel->h_dev_ptrs[stream_index]);
        channel->h_dev_ptrs[stream_index] = nullptr;
      }
    }
  }
}

void ChdrConverterOpRx::compute(
        InputContext& op_input,
        OutputContext& op_output,
        ExecutionContext& context) {
  if (use_single_channel_fast_path_) {
    auto channel = channel_list.front();

    free_bufs_and_emit_arrays(op_output, channel);

    if (static_cast<int>(channel->out_q.size()) >= max_inflight_batches()) {
      channel->backlog_events++;
      channel->periodic_summary_backlog_events++;
      HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
      HOLOSCAN_LOG_ERROR(
          "CHDR backlog state ch={} out_q_depth={} max_out_q_depth={} aggr_pkts_recv={} queued_batches={} emitted_batches={} refcounted_bursts={} oldest_release_ms={:.3f}",
          channel->channel_num,
          channel->out_q.size(),
          channel->max_out_q_depth,
          channel->aggr_pkts_recv,
          channel->completed_batches_queued,
          channel->completed_batches_emitted,
          channel->burst_refcounts.size(),
          channel->out_q.empty() ? 0.0 : elapsed_ms_since(channel->out_q.front().queued_ns));
      cudaStreamSynchronize(channel->streams[channel->cur_idx]);
      free_bufs_and_emit_arrays(op_output, channel);
    }

    BurstParams* burst = nullptr;
    auto status = get_rx_burst(&burst, single_channel_port_id_, single_channel_queue_id_);
    channel->periodic_summary_last_ns = steady_time_ns();
    if (status == Status::SUCCESS) {
      process_channel_data(op_output, burst, 0);
    } else {
      channel->empty_rx_polls++;
      channel->periodic_summary_empty_polls++;
      if (channel->aggr_pkts_recv > 0) {
        const double idle_ms = elapsed_ms_since(channel->last_receive_ns);
        if (idle_ms >= static_cast<double>(partial_batch_drop_timeout_ms_.get())) {
          flush_partial_batch(channel, "idle-timeout");
        }
      }
    }
    maybe_log_channel_summary(channel);
    return;
  }

  const auto num_rx_sources = rx_sources_.size();

  // Drain any completed batches first so ANO buffers can be freed as early as possible.
  // We reset the operator metadata before each emit so each message carries the correct
  // channel_number even when more than one completed batch is emitted in a single compute.
  uint16_t preferred_q = 0;
  size_t preferred_depth = 0;
  for (uint16_t source_index = 0; source_index < num_rx_sources; source_index++) {
    const auto& rx_source = rx_sources_.at(source_index);
    const auto channel = channel_list.at(rx_source.fallback_channel);
    if (channel->out_q.size() > preferred_depth) {
      preferred_depth = channel->out_q.size();
      preferred_q = source_index;
    }
  }

  std::vector<uint8_t> emitted_this_tick(num_rx_sources, 0);
  for (uint16_t offset = 0; offset < num_rx_sources; offset++) {
    const uint16_t source_index = (preferred_q + offset) % num_rx_sources;
    const auto& rx_source = rx_sources_.at(source_index);
    auto channel = channel_list.at(rx_source.fallback_channel);
    emitted_this_tick[source_index] = free_bufs_and_emit_arrays(op_output, channel) ? 1 : 0;
  }

  for (uint16_t offset = 0; offset < num_rx_sources; offset++) {
    const uint16_t source_index = (preferred_q + offset) % num_rx_sources;
    const auto& rx_source = rx_sources_.at(source_index);
    auto channel = channel_list.at(rx_source.fallback_channel);
    if (static_cast<int>(channel->out_q.size()) >= max_inflight_batches()) {
      channel->backlog_events++;
      channel->periodic_summary_backlog_events++;
      HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
      HOLOSCAN_LOG_ERROR(
          "CHDR backlog state ch={} interface={} queue_id={} out_q_depth={} max_out_q_depth={} aggr_pkts_recv={} queued_batches={} emitted_batches={} refcounted_bursts={} oldest_release_ms={:.3f}",
          channel->channel_num,
          rx_source.interface_name,
          rx_source.queue_id,
          channel->out_q.size(),
          channel->max_out_q_depth,
          channel->aggr_pkts_recv,
          channel->completed_batches_queued,
          channel->completed_batches_emitted,
          channel->burst_refcounts.size(),
          channel->out_q.empty() ? 0.0 : elapsed_ms_since(channel->out_q.front().queued_ns));
      cudaStreamSynchronize(channel->streams[channel->cur_idx]);
      if (!emitted_this_tick[source_index]) {
        emitted_this_tick[source_index] = free_bufs_and_emit_arrays(op_output, channel) ? 1 : 0;
      }
    }
  }


  BurstParams *burst;
  for (uint16_t source_index = 0; source_index < num_rx_sources; source_index++) {
    const auto& rx_source = rx_sources_.at(source_index);
    // If there's new data, start processing it
    auto status = get_rx_burst(&burst, rx_source.port_id, rx_source.queue_id);
    auto channel = channel_list.at(rx_source.fallback_channel);
    channel->periodic_summary_last_ns = steady_time_ns();
    if (status == Status::SUCCESS) {
      const uint16_t resolved_channel = use_flow_id_routing_
                                             ? resolve_channel_from_burst_flow(
                                                   burst,
                                                   rx_source.fallback_channel,
                                                   num_channels_.get())
                                             : rx_source.fallback_channel;
      process_channel_data(op_output, burst, resolved_channel);
    } else {
      channel->empty_rx_polls++;
      channel->periodic_summary_empty_polls++;
      if (channel->aggr_pkts_recv > 0) {
        const double idle_ms = elapsed_ms_since(channel->last_receive_ns);
        if (idle_ms >= static_cast<double>(partial_batch_drop_timeout_ms_.get())) {
          flush_partial_batch(channel, "idle-timeout");
        }
      }
    }
    maybe_log_channel_summary(channel);
  }
}

void ChdrConverterOpRx::process_channel_data(
        OutputContext& op_output,
        BurstParams *burst,
        uint16_t channel_num) {
  auto channel = channel_list.at(channel_num);

  const int burst_packets = get_num_packets(burst);
  const uint64_t receive_ts_ns = steady_time_ns();
  if (channel->periodic_summary_start_ns == 0) {
    channel->periodic_summary_start_ns = receive_ts_ns;
  }
  channel->periodic_summary_last_ns = receive_ts_ns;
  channel->rx_bursts_received++;
  channel->periodic_summary_bursts++;
  channel->periodic_summary_packets += static_cast<uint64_t>(std::max(0, burst_packets));
  channel->max_burst_packets = std::max<uint64_t>(channel->max_burst_packets,
                                                  static_cast<uint64_t>(std::max(0, burst_packets)));

  if (!channel->first_receive_logged && burst_packets > 0) {
    channel->first_receive_logged = true;
    channel->first_receive_ns = receive_ts_ns;
    HOLOSCAN_LOG_INFO("Begin receiving samples on channel {} (first burst packets={})",
                      channel->channel_num,
                      burst_packets);
  }
  if (burst_packets > 0) {
    channel->last_receive_ns = receive_ts_ns;
  }

  // Log packet details for debugging
  if (log_packets_) {
    HOLOSCAN_LOG_INFO("Processing burst on channel {} (stream {}) with {} packets",
                    channel->channel_num, channel->cur_idx, get_num_packets(burst));
    int p = 0;
    uint16_t length0 = get_segment_packet_length(burst, 0, p);
    uint16_t length1 = get_segment_packet_length(burst, 1, p);
    uint16_t length2 = get_segment_packet_length(burst, 2, p);
    HOLOSCAN_LOG_INFO("Segment 0 length: {}, Segment 1 length: {}, Segment 2 length: {}",
                      length0, length1, length2);
    auto ptr0 = get_segment_packet_ptr(burst, 0, p);
    auto ptr1 = get_segment_packet_ptr(burst, 1, p);
    auto ptr2 = get_segment_packet_ptr(burst, 2, p);
    HOLOSCAN_LOG_INFO("Segment 0 ptr: {}, Segment 1 ptr: {}, Segment 2 ptr: {}",
                      (void*)ptr0, (void*)ptr1, (void*)ptr2);
    // print bytes for each segment
    std::ostringstream oss;
    oss << "Segment '0' bytes: ";
    for (int i = 0; i < length0; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(((uint8_t*)ptr0)[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
    oss.str("");
    oss << "Segment '1' bytes: ";
    for (int i = 0; i < length1; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(((uint8_t*)ptr1)[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
    // copy from device to host memory
    uint8_t* host_buf = nullptr;
    cudaMallocHost((void**)&host_buf, length2);
    cudaMemcpy(host_buf, ptr2, length2, cudaMemcpyDeviceToHost);
    oss.str("");
    oss << "Segment '2' bytes: ";
    for (int i = 0; i < length2; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(host_buf[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
  }
  // End packet logging 

  int packet_offset = 0;
  while (packet_offset < burst_packets) {
    if (channel->aggr_pkts_recv == num_packets_per_batch) {
      queue_completed_batch(channel, num_packets_per_batch, false);
    }

    const uint32_t remaining_capacity = num_packets_per_batch - channel->aggr_pkts_recv;
    const int packets_to_copy = std::min<int>(remaining_capacity, burst_packets - packet_offset);
    if (packets_to_copy <= 0) {
      HOLOSCAN_LOG_ERROR("Unable to copy packets from CHDR burst on channel {}", channel->channel_num);
      exit(1);
    }

    if (channel->cur_msg.num_batches >= MAX_ANO_BATCHES) {
      HOLOSCAN_LOG_ERROR("Exceeded MAX_ANO_BATCHES while accumulating CHDR burst references on channel {}",
                         channel->channel_num);
      exit(1);
    }

    retain_burst_ref(channel, burst);
    channel->cur_msg.msg[channel->cur_msg.num_batches++] = burst;

    uint64_t ttl_bytes_in_cur_chunk = 0;
    for (int p = 0; p < packets_to_copy; p++) {
      const int burst_packet_idx = packet_offset + p;
      channel->h_dev_ptrs[channel->cur_idx][channel->aggr_pkts_recv + p]
          = get_segment_packet_ptr(burst, 2, burst_packet_idx);
      ttl_bytes_in_cur_chunk += get_segment_packet_length(burst, 0, burst_packet_idx)
          + get_segment_packet_length(burst, 1, burst_packet_idx)
          + get_segment_packet_length(burst, 2, burst_packet_idx);
    }

    channel->ttl_bytes_recv += ttl_bytes_in_cur_chunk;
    channel->aggr_pkts_recv += packets_to_copy;
    packet_offset += packets_to_copy;

    if (channel->aggr_pkts_recv == num_packets_per_batch) {
      queue_completed_batch(channel, num_packets_per_batch, false);
    }
  }
}

void ChdrConverterOpRx::stop() {
  adv_net_shutdown();
  // Downstream operators can still hold queued messages that reference these
  // CUDA streams while the graph is unwinding. Releasing them here can leave
  // late-stage kernels with invalid resource handles during shutdown.

  HOLOSCAN_LOG_INFO("ChdrConverterOpRx exit report:");
  for (uint16_t channel_num = 0; channel_num < num_channels_.get(); channel_num++) {
    auto channel = channel_list.at(channel_num);
  const uint64_t duration_ns =
    (channel->first_receive_ns != 0 && channel->last_receive_ns >= channel->first_receive_ns)
      ? (channel->last_receive_ns - channel->first_receive_ns)
      : 0;
  const double duration_s = static_cast<double>(duration_ns) / 1.0e9;
  const double processed_samples =
    static_cast<double>(channel->ttl_pkts_recv) * static_cast<double>(num_complex_samples_per_packet_.get());
  const double avg_msps =
    (duration_s > 0.0) ? (processed_samples / duration_s / 1.0e6) : 0.0;
  const double avg_gbps =
    (duration_s > 0.0)
      ? (processed_samples * sizeof(int16_t) * 2 * 8 / duration_s / 1.0e9)
      : 0.0;
    HOLOSCAN_LOG_INFO(
        "\n"
        "------- CH {} --------\n"
        "   Processed bytes: {}\n"
        " Processed packets: {}\n"
    " Active receive sec: {:.3f}\n"
    " Average throughput: {:.2f} MSps ({:.2f} Gbps)\n"
        " Completed batches queued: {}\n"
        "Completed batches emitted: {}\n"
        "        RX bursts received: {}\n"
        "           Empty RX polls: {}\n"
        "     Partial batch drops: {}\n"
        "         Backlog events: {}\n"
        "       Max out_q depth: {}\n"
        "      Max burst packets: {}\n"
        " Mean release dwell ms: {:.3f}\n"
        "  Max release dwell ms: {:.3f}\n",
        channel->channel_num,
        channel->ttl_bytes_recv,
        channel->ttl_pkts_recv,
        duration_s,
        avg_msps,
        avg_gbps,
        channel->completed_batches_queued,
        channel->completed_batches_emitted,
          channel->rx_bursts_received,
          channel->empty_rx_polls,
        channel->timeout_like_partial_drains,
        channel->backlog_events,
        channel->max_out_q_depth,
          channel->max_burst_packets,
        channel->release_samples == 0 ? 0.0 : channel->total_release_latency_ms / static_cast<double>(channel->release_samples),
        channel->max_release_latency_ms);
  }
  holoscan::Operator::stop();
}
}  // namespace holoscan::ops

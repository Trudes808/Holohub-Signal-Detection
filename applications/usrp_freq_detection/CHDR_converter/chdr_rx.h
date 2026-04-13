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

#ifndef CHDR_CONVERTER_CHDR_RX_H
#define CHDR_CONVERTER_CHDR_RX_H

#include "holoscan/holoscan.hpp"
#include "matx.h"
#include "advanced_network/common.h"

#include <unordered_map>

using namespace holoscan::advanced_network;
using namespace matx;
using complex = cuda::std::complex<float>;

namespace holoscan::ops {

class ChdrConverterOpRx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ChdrConverterOpRx)

  ChdrConverterOpRx() = default;

  ~ChdrConverterOpRx() {
    shutdown();
    print_stats();
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  static constexpr int num_concurrent  = 4;   // Number of concurrent batches processing
  static constexpr int MAX_ANO_BATCHES = 20;  // Batches from ANO for one app batch

  Parameter<uint16_t> num_complex_samples_per_packet_;
  Parameter<uint16_t> num_packets_per_fft_;
  Parameter<uint16_t> num_ffts_per_batch_;
  Parameter<uint16_t> num_simul_batches_;
  Parameter<uint16_t> num_channels_;
  Parameter<std::string> interface_name_;
  Parameter<bool> log_data_;
  Parameter<bool> log_packets_;
  int port_id_;
  uint32_t num_packets_per_batch;

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<BurstParams *, MAX_ANO_BATCHES> msg;
    int num_batches;
    int batch_idx;
    cudaStream_t stream;
    cudaEvent_t evt;
  };

  struct Channel {
    uint16_t channel_num;
    int cur_idx = 0;
    tensor_t<complex, 3> rf_data;
    std::array<void **, num_concurrent> h_dev_ptrs;
    std::array<cudaStream_t, num_concurrent> streams;
    std::array<cudaEvent_t, num_concurrent> events;
    RxMsg cur_msg{};
    std::queue<RxMsg> out_q;
    std::unordered_map<BurstParams*, uint32_t> burst_refcounts;
    uint64_t ttl_bytes_recv = 0;
    uint64_t ttl_pkts_recv = 0;
    uint64_t aggr_pkts_recv = 0;
  };

  std::vector<std::shared_ptr<struct Channel>> channel_list;

  std::optional<RxMsg> free_buf(std::shared_ptr<struct Channel> channel);
  bool free_bufs_and_emit_arrays(OutputContext& op_output, std::shared_ptr<struct Channel> channel);
  void retain_burst_ref(std::shared_ptr<struct Channel> channel, BurstParams* burst);
  void release_burst_ref(std::shared_ptr<struct Channel> channel, BurstParams* burst);
  void queue_completed_batch(std::shared_ptr<struct Channel> channel);
  void process_channel_data(
          OutputContext& op_output,
          BurstParams *burst,
          uint16_t channel_num);
  int max_inflight_batches() const;
};  // ChdrConverterOpRx

}  // namespace holoscan::ops


#endif /* CHDR_CONVERTER_CHDR_RX_H */

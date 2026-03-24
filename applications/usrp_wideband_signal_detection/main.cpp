// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "../usrp_freq_detection/CHDR_converter/chdr_rx.h"
#include <dinov3_signal_detector.hpp>
#include <fft.hpp>
#include <spectrogram.hpp>

class LogOp: public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LogOp)

  using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

  LogOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<in_t>("in");
    spec.param(num_channels_,
               "num_channels",
               "Number of Channels",
               "The number of RF channels being processed.",
               1);
    spec.param(log_interval_,
               "log_interval",
               "Log Interval",
               "Interval in seconds to log the data rate statistics.",
               5);
  }

  void initialize() {
    holoscan::Operator::initialize();
    total_samples_.resize(num_channels_, 0);
    start_.resize(num_channels_, std::chrono::steady_clock::now());
    elapsed_.resize(num_channels_, std::chrono::steady_clock::duration::zero());
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto input = op_input.receive<in_t>("in").value();
    auto tensor = std::get<0>(input);

    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);

    auto now = std::chrono::steady_clock::now();
    auto interval = now - start_[channel_num];
    start_[channel_num] = now;

    auto num_samples = tensor.Size(0) * tensor.Size(1);
    auto num_bits = num_samples * sizeof(int16_t) * 2 * 8;

    total_samples_[channel_num] += num_samples;
    elapsed_[channel_num] += interval;

    auto seconds = std::chrono::duration<double>(elapsed_[channel_num]).count();
    if (total_samples_[channel_num] > 0 && seconds >= log_interval_) {
      auto duration = std::chrono::duration<double>(interval).count();
      HOLOSCAN_LOG_INFO("Processed {} samples from channel {} at {:.2f} MSps ({:.2f} Gbps)",
                        total_samples_[channel_num],
                        channel_num,
                        num_samples / duration / 1e6,
                        num_bits / duration / 1e9);
      total_samples_[channel_num] = 0;
      elapsed_[channel_num] = std::chrono::steady_clock::duration::zero();
    }
  }

 private:
  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> log_interval_;
  std::vector<int64_t> total_samples_;
  std::vector<std::chrono::steady_clock::time_point> start_;
  std::vector<std::chrono::steady_clock::duration> elapsed_;
};

class UsrpWidebandSignalDetectionPipeline : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
    if (adv_net_init(adv_net_config) != Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
      exit(1);
    }
    HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");

    auto chdrConverterOp = make_operator<ops::ChdrConverterOpRx>("chdrConverterOp", from_config("chdr_converter"));

    auto fftOp = make_operator<ops::FFT>("fftOp", from_config("fft"));

    auto spectrogramOp = make_operator<ops::Spectrogram>("spectrogramOp", from_config("spectrogram"));

    auto dinoV3SignalDetectorOp = make_operator<ops::DinoV3SignalDetector>(
        "dinoV3SignalDetectorOp",
        from_config("dinov3_signal_detector"));

    auto logOp = make_operator<LogOp>(
        "logOp",
        from_config("logger"),
        make_condition<CountCondition>(from_config("num_runs").as<int64_t>()));

    add_operator(chdrConverterOp);
    add_operator(fftOp);
    add_operator(spectrogramOp);
    add_operator(dinoV3SignalDetectorOp);
    add_operator(logOp);

    add_flow(chdrConverterOp, fftOp);
    add_flow(fftOp, spectrogramOp);
    add_flow(spectrogramOp, dinoV3SignalDetectorOp);
    add_flow(fftOp, logOp);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<UsrpWidebandSignalDetectionPipeline>();

  app->enable_metadata(true);

  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
    return -1;
  }

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/" + std::string(argv[1]);

  if (!std::filesystem::exists(config_path)) {
    HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist", static_cast<std::string>(config_path));
    return -1;
  }

  app->config(config_path);

  app->scheduler(
      app->make_scheduler<holoscan::EventBasedScheduler>("event-based-scheduler", app->from_config("scheduler")));

  app->run();

  shutdown();
  return 0;
}

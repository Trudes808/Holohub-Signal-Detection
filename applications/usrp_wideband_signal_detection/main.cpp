// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "../usrp_freq_detection/CHDR_converter/chdr_rx.h"
#include "spectrogram_visualization.hpp"
#include <coherent_power_signal_detector.hpp>
#include <dinov3_signal_detector.hpp>
#include <fft.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <spectrogram.hpp>

namespace {

std::filesystem::path resolve_config_path(const char* argv0, const char* config_arg) {
  const std::filesystem::path requested(config_arg);
  const auto binary_dir = std::filesystem::canonical(argv0).parent_path();

  if (requested.is_absolute() && std::filesystem::exists(requested)) {
    return requested;
  }

  if (std::filesystem::exists(requested)) {
    return std::filesystem::absolute(requested);
  }

  const auto from_binary_dir = binary_dir / requested;
  if (std::filesystem::exists(from_binary_dir)) {
    return from_binary_dir;
  }

  const auto from_source_dir = std::filesystem::path(USRP_WIDEBAND_APP_SOURCE_DIR) / requested;
  if (std::filesystem::exists(from_source_dir)) {
    return from_source_dir;
  }

  return from_binary_dir;
}

}  // namespace

class LogOp: public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LogOp)

  using in_t = holoscan::ops::in_t;

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

    const bool enable_spectrogram = from_config("pipeline.enable_spectrogram").as<bool>();
    const bool enable_detector = from_config("pipeline.enable_detector").as<bool>();
    const std::string detector_type = from_config("pipeline.detector_type").as<std::string>();
    const bool log_from_spectrogram = from_config("pipeline.log_from_spectrogram").as<bool>();

    if (enable_detector && !enable_spectrogram) {
      HOLOSCAN_LOG_ERROR("pipeline.enable_detector=true requires pipeline.enable_spectrogram=true");
      exit(1);
    }

    if (enable_detector && detector_type != "dinov3" && detector_type != "coherent_power") {
      HOLOSCAN_LOG_ERROR("Unsupported pipeline.detector_type='{}'. Expected 'dinov3' or 'coherent_power'.",
                         detector_type);
      exit(1);
    }

    std::shared_ptr<holoscan::Operator> spectrogramOp;
    std::shared_ptr<holoscan::Operator> detectorOp;
    if (enable_spectrogram) {
      spectrogramOp = make_operator<ops::Spectrogram>("spectrogramOp", from_config("spectrogram"));
    }
    if (enable_detector) {
      if (detector_type == "dinov3") {
        detectorOp = make_operator<ops::DinoV3SignalDetector>("dinoV3SignalDetectorOp",
                                                               from_config("dinov3_signal_detector"));
      } else {
        detectorOp = make_operator<ops::CoherentPowerSignalDetector>(
            "coherentPowerSignalDetectorOp",
            from_config("coherent_power_signal_detector"));
      }
    }

    const bool enable_visualization = from_config("visualization.enable").as<bool>();
    if (enable_visualization && !enable_spectrogram) {
      HOLOSCAN_LOG_ERROR("visualization.enable=true requires pipeline.enable_spectrogram=true");
      exit(1);
    }
    std::shared_ptr<holoscan::Operator> spectrogramVisualizerOp;
    std::shared_ptr<holoscan::Operator> holovizOp;
    if (enable_visualization) {
      const auto tensor_name = from_config("visualization.renderer.tensor_name").as<std::string>();
      spectrogramVisualizerOp = make_operator<ops::SpectrogramToHolovizOp>(
        "spectrogramVisualizerOp",
        from_config("visualization.renderer"));
      holovizOp = make_operator<ops::HolovizOp>(
        "holovizOp",
        from_config("visualization.holoviz"),
        holoscan::Arg("tensors") = ops::make_spectrogram_input_specs(tensor_name));
    }

    auto logOp = make_operator<LogOp>(
        "logOp",
        from_config("logger"),
        make_condition<CountCondition>(from_config("num_runs").as<int64_t>()));

    add_operator(chdrConverterOp);
    add_operator(fftOp);
    if (enable_spectrogram) {
      add_operator(spectrogramOp);
    }
    if (enable_detector) {
      add_operator(detectorOp);
    }
    if (enable_visualization) {
      add_operator(spectrogramVisualizerOp);
      add_operator(holovizOp);
    }
    add_operator(logOp);

    add_flow(chdrConverterOp, fftOp);
    if (enable_spectrogram) {
      add_flow(fftOp, spectrogramOp);
    }
    if (enable_detector) {
      add_flow(spectrogramOp, detectorOp);
    }
    if (enable_visualization) {
      add_flow(spectrogramOp, spectrogramVisualizerOp);
      add_flow(spectrogramVisualizerOp, holovizOp, {{"outputs", "receivers"}});
    }
    if (log_from_spectrogram) {
      if (!enable_spectrogram) {
        HOLOSCAN_LOG_ERROR("pipeline.log_from_spectrogram=true requires pipeline.enable_spectrogram=true");
        exit(1);
      }
      add_flow(spectrogramOp, logOp);
    } else {
      add_flow(fftOp, logOp);
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<UsrpWidebandSignalDetectionPipeline>();

  app->enable_metadata(true);

  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
    return -1;
  }

  auto config_path = resolve_config_path(argv[0], argv[1]);

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

// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "../usrp_freq_detection/CHDR_converter/chdr_rx.h"
#include "spectrogram_visualization.hpp"
#include <algorithm>
#include <limits>
#include <optional>
#include <coherent_power_signal_detector.hpp>
#include <cuda_dino_detector.hpp>
#include <dinov3_signal_detector.hpp>
#include <fft.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <spectrogram.hpp>

#ifdef USRP_WIDEBAND_HAS_NVML
#include <nvml.h>
#endif

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
    auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{8});
    input_port.conditions().emplace_back(
        holoscan::ConditionType::kMessageAvailable,
        std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
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
    gpu_util_sum_.resize(num_channels_, 0.0);
    gpu_util_samples_.resize(num_channels_, 0);
    gpu_util_min_.resize(num_channels_, std::numeric_limits<unsigned int>::max());
    gpu_util_max_.resize(num_channels_, 0);
    gpu_sample_start_.resize(num_channels_, std::chrono::steady_clock::time_point {});
    last_gpu_sample_.resize(num_channels_, std::chrono::steady_clock::time_point {});

#ifdef USRP_WIDEBAND_HAS_NVML
    const auto init_result = nvmlInit_v2();
    if (init_result == NVML_SUCCESS) {
      nvml_device_count_ = 0;
      if (nvmlDeviceGetCount_v2(&nvml_device_count_) == NVML_SUCCESS && nvml_device_count_ > 0) {
        nvmlDevice_t handle = nullptr;
        if (nvmlDeviceGetHandleByIndex_v2(0, &handle) == NVML_SUCCESS) {
          nvml_device_ = handle;
        }
      }
      if (!nvml_device_.has_value()) {
        nvmlShutdown();
      }
    }
#endif
  }

  void stop() override {
#ifdef USRP_WIDEBAND_HAS_NVML
    if (nvml_device_.has_value()) {
      nvmlShutdown();
      nvml_device_.reset();
    }
#endif
    holoscan::Operator::stop();
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

    maybe_sample_gpu_util(channel_num, now);

    auto seconds = std::chrono::duration<double>(elapsed_[channel_num]).count();
    if (total_samples_[channel_num] > 0 && seconds >= log_interval_) {
      const double samples_per_second = static_cast<double>(total_samples_[channel_num]) / seconds;
      const double bits_per_second = static_cast<double>(total_samples_[channel_num]) * sizeof(int16_t) * 2 * 8 / seconds;
      if (gpu_util_samples_[channel_num] > 0) {
        const double mean_gpu_util = gpu_util_sum_[channel_num] / static_cast<double>(gpu_util_samples_[channel_num]);
        const unsigned int min_gpu_util = gpu_util_min_[channel_num] == std::numeric_limits<unsigned int>::max() ? 0U : gpu_util_min_[channel_num];
        HOLOSCAN_LOG_INFO("Processed {} samples from channel {} at {:.2f} MSps ({:.2f} Gbps) gpu_util_pct(mean={:.2f},min={},max={})",
                          total_samples_[channel_num],
                          channel_num,
                          samples_per_second / 1e6,
                          bits_per_second / 1e9,
                          mean_gpu_util,
                          min_gpu_util,
                          gpu_util_max_[channel_num]);
      } else {
        HOLOSCAN_LOG_INFO("Processed {} samples from channel {} at {:.2f} MSps ({:.2f} Gbps)",
                          total_samples_[channel_num],
                          channel_num,
                          samples_per_second / 1e6,
                          bits_per_second / 1e9);
      }
      total_samples_[channel_num] = 0;
      elapsed_[channel_num] = std::chrono::steady_clock::duration::zero();
      gpu_util_sum_[channel_num] = 0.0;
      gpu_util_samples_[channel_num] = 0;
      gpu_util_min_[channel_num] = std::numeric_limits<unsigned int>::max();
      gpu_util_max_[channel_num] = 0;
      gpu_sample_start_[channel_num] = std::chrono::steady_clock::time_point {};
    }
  }

 private:
  void maybe_sample_gpu_util(uint16_t channel_num, const std::chrono::steady_clock::time_point& now) {
    if (channel_num >= gpu_util_sum_.size()) {
      return;
    }
#ifdef USRP_WIDEBAND_HAS_NVML
    if (!nvml_device_.has_value()) {
      return;
    }
    constexpr auto kGpuSamplePeriod = std::chrono::milliseconds(250);
    if (last_gpu_sample_[channel_num] != std::chrono::steady_clock::time_point {} &&
        now - last_gpu_sample_[channel_num] < kGpuSamplePeriod) {
      return;
    }
    nvmlUtilization_t utilization {};
    if (nvmlDeviceGetUtilizationRates(*nvml_device_, &utilization) != NVML_SUCCESS) {
      return;
    }
    last_gpu_sample_[channel_num] = now;
    if (gpu_sample_start_[channel_num] == std::chrono::steady_clock::time_point {}) {
      gpu_sample_start_[channel_num] = now;
    }
    gpu_util_sum_[channel_num] += static_cast<double>(utilization.gpu);
    gpu_util_samples_[channel_num] += 1;
    gpu_util_min_[channel_num] = std::min(gpu_util_min_[channel_num], utilization.gpu);
    gpu_util_max_[channel_num] = std::max(gpu_util_max_[channel_num], utilization.gpu);
#else
    static_cast<void>(channel_num);
    static_cast<void>(now);
#endif
  }

  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> log_interval_;
  std::vector<int64_t> total_samples_;
  std::vector<std::chrono::steady_clock::time_point> start_;
  std::vector<std::chrono::steady_clock::duration> elapsed_;
  std::vector<double> gpu_util_sum_;
  std::vector<uint64_t> gpu_util_samples_;
  std::vector<unsigned int> gpu_util_min_;
  std::vector<unsigned int> gpu_util_max_;
  std::vector<std::chrono::steady_clock::time_point> gpu_sample_start_;
  std::vector<std::chrono::steady_clock::time_point> last_gpu_sample_;

#ifdef USRP_WIDEBAND_HAS_NVML
  std::optional<nvmlDevice_t> nvml_device_;
  unsigned int nvml_device_count_ = 0;
#endif
};

class DropOp: public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DropOp)

  using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

  DropOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{8});
    input_port.conditions().emplace_back(
        holoscan::ConditionType::kMessageAvailable,
        std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    static_cast<void>(op_input.receive<in_t>("in"));
  }
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
    const int pipeline_channels = std::max(1, from_config("chdr_converter.num_channels").as<int>());

    const bool enable_spectrogram = from_config("pipeline.enable_spectrogram").as<bool>();
    const bool enable_detector = from_config("pipeline.enable_detector").as<bool>();
    const std::string detector_type = from_config("pipeline.detector_type").as<std::string>();
    const bool log_from_spectrogram = from_config("pipeline.log_from_spectrogram").as<bool>();
    const bool enable_visualization = from_config("visualization.enable").as<bool>();
    const bool spectrogram_save_enabled =
      enable_spectrogram ? from_config("spectrogram.enable_save").as<bool>() : false;
    const bool spectrogram_tensor_save_enabled =
      enable_spectrogram ? from_config("spectrogram.enable_tensor_save").as<bool>() : false;
    const bool bypass_spectrogram_passthrough =
        enable_spectrogram && enable_detector && !log_from_spectrogram && !enable_visualization &&
        !spectrogram_save_enabled && !spectrogram_tensor_save_enabled;
    const bool enable_logger_branch = !bypass_spectrogram_passthrough;
    const bool force_logger_from_spectrogram =
      enable_spectrogram && enable_logger_branch && !enable_detector && !enable_visualization;
    const bool effective_log_from_spectrogram = log_from_spectrogram || force_logger_from_spectrogram;
    const bool detector_uses_dino_style_stride =
      enable_detector && (detector_type == "dinov3" || detector_type == "cuda_dino");
    const int configured_dino_emit_stride =
      (enable_detector && detector_type == "dinov3")
        ? std::max(1, from_config("dinov3_signal_detector.emit_stride").as<int>())
      : (enable_detector && detector_type == "cuda_dino")
        ? std::max(1, from_config("cuda_dino_detector.emit_stride").as<int>())
        : 1;
    const bool enable_fft_emit_stride =
      bypass_spectrogram_passthrough && detector_uses_dino_style_stride &&
        configured_dino_emit_stride > 1;

    std::vector<std::shared_ptr<holoscan::Operator>> fftOps;
    fftOps.reserve(static_cast<size_t>(pipeline_channels));
    for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
      fftOps.push_back(make_operator<ops::FFT>(
        std::string("fftOpCh") + std::to_string(channel_index),
        from_config("fft"),
        holoscan::Arg("emit_stride") = (enable_fft_emit_stride ? configured_dino_emit_stride : 1)));
    }

    if (enable_detector && !enable_spectrogram) {
      HOLOSCAN_LOG_ERROR("pipeline.enable_detector=true requires pipeline.enable_spectrogram=true");
      exit(1);
    }

    if (enable_detector && detector_type != "dinov3" && detector_type != "cuda_dino" && detector_type != "coherent_power") {
      HOLOSCAN_LOG_ERROR("Unsupported pipeline.detector_type='{}'. Expected 'dinov3', 'cuda_dino', or 'coherent_power'.",
                         detector_type);
      exit(1);
    }

    std::vector<std::shared_ptr<holoscan::Operator>> spectrogramOps;
    std::vector<std::shared_ptr<holoscan::Operator>> dinoDetectorOps;
    std::vector<std::shared_ptr<holoscan::Operator>> cudaDinoDetectorOps;
    std::vector<std::shared_ptr<holoscan::Operator>> coherentDetectorOps;
    if (enable_spectrogram) {
      spectrogramOps.reserve(static_cast<size_t>(pipeline_channels));
      for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
        spectrogramOps.push_back(make_operator<ops::Spectrogram>(
            std::string("spectrogramOpCh") + std::to_string(channel_index),
            from_config("spectrogram")));
      }
    }
    if (enable_detector) {
      if (detector_type == "dinov3") {
        const int detector_channels = from_config("dinov3_signal_detector.num_channels").as<int>();
        if (detector_channels != pipeline_channels) {
          HOLOSCAN_LOG_ERROR("dinov3_signal_detector.num_channels={} must match chdr_converter.num_channels={} for one-to-one channel routing.",
                             detector_channels,
                             pipeline_channels);
          exit(1);
        }
        for (int channel_index = 0; channel_index < std::max(1, detector_channels); ++channel_index) {
          dinoDetectorOps.push_back(make_operator<ops::DinoV3SignalDetector>(
              std::string("dinoV3SignalDetectorOpCh") + std::to_string(channel_index),
              from_config("dinov3_signal_detector"),
              holoscan::Arg("channel_filter") = channel_index,
              holoscan::Arg("emit_stride") = (enable_fft_emit_stride ? 1 : configured_dino_emit_stride)));
        }
      } else if (detector_type == "cuda_dino") {
        const int detector_channels = from_config("cuda_dino_detector.num_channels").as<int>();
        if (detector_channels != pipeline_channels) {
          HOLOSCAN_LOG_ERROR("cuda_dino_detector.num_channels={} must match chdr_converter.num_channels={} for one-to-one channel routing.",
                             detector_channels,
                             pipeline_channels);
          exit(1);
        }
        for (int channel_index = 0; channel_index < std::max(1, detector_channels); ++channel_index) {
          cudaDinoDetectorOps.push_back(make_operator<ops::CudaDinoDetector>(
              std::string("cudaDinoDetectorOpCh") + std::to_string(channel_index),
              from_config("cuda_dino_detector"),
              holoscan::Arg("channel_filter") = channel_index,
              holoscan::Arg("emit_stride") = (enable_fft_emit_stride ? 1 : configured_dino_emit_stride)));
        }
      } else {
        const int detector_channels = from_config("coherent_power_signal_detector.num_channels").as<int>();
        if (detector_channels != pipeline_channels) {
          HOLOSCAN_LOG_ERROR("coherent_power_signal_detector.num_channels={} must match chdr_converter.num_channels={} for one-to-one channel routing.",
                             detector_channels,
                             pipeline_channels);
          exit(1);
        }
        for (int channel_index = 0; channel_index < std::max(1, detector_channels); ++channel_index) {
          coherentDetectorOps.push_back(make_operator<ops::CoherentPowerSignalDetector>(
              std::string("coherentPowerSignalDetectorOpCh") + std::to_string(channel_index),
              from_config("coherent_power_signal_detector"),
              holoscan::Arg("channel_filter") = channel_index));
        }
      }
    }

    if (enable_visualization && !enable_spectrogram) {
      HOLOSCAN_LOG_ERROR("visualization.enable=true requires pipeline.enable_spectrogram=true");
      exit(1);
    }

    if (bypass_spectrogram_passthrough) {
      HOLOSCAN_LOG_INFO(
          "Bypassing spectrogramOp in lean performance mode; detector will consume FFT output directly.");
      HOLOSCAN_LOG_INFO(
        "Disabling FFT logger branch in lean performance mode to keep the hot path single-consumer.");
      if (enable_fft_emit_stride) {
        HOLOSCAN_LOG_INFO(
            "Enabling FFT-side emit_stride={} so skipped batches are dropped before any downstream queueing.",
            configured_dino_emit_stride);
      }
    }
    if (force_logger_from_spectrogram && !log_from_spectrogram) {
      HOLOSCAN_LOG_INFO(
          "Routing logger from spectrogram output because the detector is disabled and spectrogramOp needs a downstream consumer.");
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
    std::vector<std::shared_ptr<holoscan::Operator>> logOps;
    if (enable_logger_branch) {
      logOps.reserve(static_cast<size_t>(pipeline_channels));
      for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
        logOps.push_back(make_operator<LogOp>(
            std::string("logOpCh") + std::to_string(channel_index),
            from_config("logger"),
            make_condition<CountCondition>(from_config("num_runs").as<int64_t>())));
      }
    }
    std::shared_ptr<holoscan::Operator> unusedChdrOutputDropOp;
    if (pipeline_channels < 2) {
      unusedChdrOutputDropOp = make_operator<DropOp>("unusedChdrOutputDropOp");
    }

    add_operator(chdrConverterOp);
    for (auto& op : fftOps) {
      add_operator(op);
    }
    if (enable_spectrogram && !bypass_spectrogram_passthrough) {
      for (auto& op : spectrogramOps) {
        add_operator(op);
      }
    }
    if (enable_detector) {
      if (detector_type == "coherent_power") {
        for (auto& op : coherentDetectorOps) {
          add_operator(op);
        }
      } else if (detector_type == "cuda_dino") {
        for (auto& op : cudaDinoDetectorOps) {
          add_operator(op);
        }
      } else {
        for (auto& op : dinoDetectorOps) {
          add_operator(op);
        }
      }
    }
    if (enable_visualization) {
      add_operator(spectrogramVisualizerOp);
      add_operator(holovizOp);
    }
    for (auto& op : logOps) {
      add_operator(op);
    }
    if (unusedChdrOutputDropOp) {
      add_operator(unusedChdrOutputDropOp);
    }

    for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
      const char* chdr_port = channel_index == 0 ? "out0" : "out1";
      auto& fftOp = fftOps[static_cast<size_t>(channel_index)];
      add_flow(chdrConverterOp, fftOp, {{chdr_port, "in"}});

      std::shared_ptr<holoscan::Operator> detector_source = fftOp;
      if (enable_spectrogram && !bypass_spectrogram_passthrough) {
        auto& spectrogramOp = spectrogramOps[static_cast<size_t>(channel_index)];
        add_flow(fftOp, spectrogramOp);
        detector_source = spectrogramOp;

        if (enable_visualization) {
          //add_flow(spectrogramOp, spectrogramVisualizerOp);
          add_flow(spectrogramOp, spectrogramVisualizerOp, {{"out", "in"}});
        }
      }

      if (enable_detector) {
        if (detector_type == "coherent_power") {
          add_flow(detector_source, coherentDetectorOps[static_cast<size_t>(channel_index)]);
        } else if (detector_type == "cuda_dino") {
          add_flow(detector_source, cudaDinoDetectorOps[static_cast<size_t>(channel_index)]);
        } else {
          add_flow(detector_source, dinoDetectorOps[static_cast<size_t>(channel_index)]);
        }
      }

      if (enable_logger_branch && effective_log_from_spectrogram) {
        if (!enable_spectrogram) {
          HOLOSCAN_LOG_ERROR("pipeline.log_from_spectrogram=true requires pipeline.enable_spectrogram=true");
          exit(1);
        }
        add_flow(detector_source, logOps[static_cast<size_t>(channel_index)]);
      } else if (enable_logger_branch) {
        add_flow(fftOp, logOps[static_cast<size_t>(channel_index)]);
      }
    }

    if (unusedChdrOutputDropOp) {
      add_flow(chdrConverterOp, unusedChdrOutputDropOp, {{"out1", "in"}});
    }

    if (enable_visualization) {
      add_flow(spectrogramVisualizerOp, holovizOp, {{"outputs", "receivers"}});
    }
    if (enable_visualization && enable_detector && detector_type == "coherent_power") {
      for (int ch = 0; ch < pipeline_channels; ++ch) {
        add_flow(coherentDetectorOps[static_cast<size_t>(ch)],
                 spectrogramVisualizerOp,
                 {{"mask_out", "mask_in"}});
      }
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<UsrpWidebandSignalDetectionPipeline>();

  app->enable_metadata(true);

  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} [config-path]", argv[0]);
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

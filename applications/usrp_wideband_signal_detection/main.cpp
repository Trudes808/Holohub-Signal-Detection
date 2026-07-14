// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "../usrp_freq_detection/CHDR_converter/chdr_rx.h"
#include "fft_runtime_config.hpp"
#include "sigmf_file_sink.hpp"
#include "signal_snipper.hpp"
#include "spectrogram_visualization.hpp"
#include "advanced_network/common.h"
#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <coherent_power_signal_detector.hpp>
#include <cuda_dino_detector.hpp>
#include <fft.hpp>
#include <gxf/core/gxf.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/imgui.h>
#include <spectrogram.hpp>

#ifdef USRP_WIDEBAND_HAS_NVML
#include <nvml.h>
#endif

namespace {

std::shared_ptr<holoscan::BooleanCondition> g_visualization_shutdown_term;
std::shared_ptr<holoscan::BooleanCondition> g_pipeline_shutdown_term;

void request_visualization_shutdown() {
  auto shutdown_term = g_visualization_shutdown_term;
  if (shutdown_term) {
    shutdown_term->disable_tick();
  }
}

void request_pipeline_shutdown() {
  auto shutdown_term = g_pipeline_shutdown_term;
  if (shutdown_term) {
    shutdown_term->disable_tick();
  }
}

void request_graceful_shutdown(gxf_context_t app_context) {
  static_cast<void>(app_context);
  std::fprintf(stderr, "[copilot-probe] request_graceful_shutdown()\n");
  std::fflush(stderr);
  HOLOSCAN_LOG_INFO("request_graceful_shutdown()");
  holoscan::advanced_network::adv_net_request_shutdown();
  request_pipeline_shutdown();
  request_visualization_shutdown();
}

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

void reserve_adv_network_cores_for_process_threads(const holoscan::advanced_network::NetworkConfig& config) {
  std::vector<int> reserved_cores;
  reserved_cores.push_back(config.common_.master_core_);
  for (const auto& intf : config.ifs_) {
    for (const auto& queue : intf.rx_.queues_) {
      reserved_cores.push_back(std::strtol(queue.common_.cpu_core_.c_str(), nullptr, 10));
    }
    for (const auto& queue : intf.tx_.queues_) {
      reserved_cores.push_back(std::strtol(queue.common_.cpu_core_.c_str(), nullptr, 10));
    }
  }

  auto add_thread_siblings = [&](int cpu) {
    if (cpu < 0) {
      return;
    }
    std::ifstream siblings_file("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                "/topology/thread_siblings_list");
    if (!siblings_file.is_open()) {
      return;
    }

    std::string siblings;
    std::getline(siblings_file, siblings);
    std::stringstream range_stream(siblings);
    std::string token;
    while (std::getline(range_stream, token, ',')) {
      const auto dash = token.find('-');
      if (dash == std::string::npos) {
        reserved_cores.push_back(std::strtol(token.c_str(), nullptr, 10));
        continue;
      }

      const int start = std::strtol(token.substr(0, dash).c_str(), nullptr, 10);
      const int stop = std::strtol(token.substr(dash + 1).c_str(), nullptr, 10);
      for (int sibling = start; sibling <= stop; ++sibling) {
        reserved_cores.push_back(sibling);
      }
    }
  };

  const auto explicit_reserved_cores = reserved_cores;
  for (const int cpu : explicit_reserved_cores) {
    add_thread_siblings(cpu);
  }

  std::sort(reserved_cores.begin(), reserved_cores.end());
  reserved_cores.erase(std::unique(reserved_cores.begin(), reserved_cores.end()), reserved_cores.end());

  cpu_set_t allowed_cores;
  CPU_ZERO(&allowed_cores);
  const long cpu_count = ::sysconf(_SC_NPROCESSORS_ONLN);
  if (cpu_count <= 0) {
    return;
  }

  int allowed_count = 0;
  for (int cpu = 0; cpu < cpu_count; ++cpu) {
    if (!std::binary_search(reserved_cores.begin(), reserved_cores.end(), cpu)) {
      CPU_SET(cpu, &allowed_cores);
      ++allowed_count;
    }
  }

  if (allowed_count == 0) {
    HOLOSCAN_LOG_WARN("Skipping process CPU affinity isolation because advanced-network reserved all online CPUs");
    return;
  }

  const int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &allowed_cores);
  if (rc != 0) {
    HOLOSCAN_LOG_WARN("Failed to reserve advanced-network cores from process threads: pthread_setaffinity_np returned {}",
                      rc);
    return;
  }

  std::ostringstream reserved_desc;
  for (size_t index = 0; index < reserved_cores.size(); ++index) {
    if (index > 0) {
      reserved_desc << ',';
    }
    reserved_desc << reserved_cores[index];
  }
  HOLOSCAN_LOG_INFO("Reserved advanced-network cores and SMT siblings [{}] from non-DPDK process threads",
                    reserved_desc.str());
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

class LoggingHolovizOp : public holoscan::ops::HolovizOp {
 public:
      using holoscan::ops::HolovizOp::HolovizOp;

  void initialize() override {
    std::fprintf(stderr, "[copilot-probe] LoggingHolovizOp::initialize()\n");
    std::fflush(stderr);
    HOLOSCAN_LOG_INFO("LoggingHolovizOp::initialize()");
    holoscan::ops::HolovizOp::initialize();
  }

  void stop() override {
    std::fprintf(stderr, "[copilot-probe] LoggingHolovizOp::stop() begin\n");
    std::fflush(stderr);
    HOLOSCAN_LOG_INFO("LoggingHolovizOp::stop() begin");
    holoscan::ops::HolovizOp::stop();
    std::fprintf(stderr, "[copilot-probe] LoggingHolovizOp::stop() complete\n");
    std::fflush(stderr);
    HOLOSCAN_LOG_INFO("LoggingHolovizOp::stop() complete");
  }
};

class UsrpWidebandSignalDetectionPipeline : public holoscan::Application {
 public:
  void layer_callback(const std::vector<holoscan::gxf::Entity>&) {
    holoscan::viz::BeginImGuiLayer();
    holoscan::ops::render_visualization_ui_overlay();
    holoscan::viz::EndLayer();
  }

  void compose() override {
    using namespace holoscan;

    auto adv_net_config = from_config("advanced_network").as<NetworkConfig>();
    if (adv_net_init(adv_net_config) != Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager");
      exit(1);
    }
    HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");
    reserve_adv_network_cores_for_process_threads(adv_net_config);

    auto pipeline_shutdown_term = make_condition<BooleanCondition>("pipeline_shutdown_term", true);
    g_pipeline_shutdown_term = pipeline_shutdown_term;
    HOLOSCAN_LOG_INFO("Configured pipeline_shutdown_term for chdrConverterOp");

    // Launch-time overrides so a new radio rate/center flows through the whole pipeline WITHOUT
    // editing the config. Offline already derives these from the SigMF; for live/loopback the run
    // wrapper (or user) can set USRP_SAMPLE_RATE_HZ / USRP_CENTER_FREQ_HZ (e.g. from the replayed
    // SigMF's core:sample_rate/core:frequency, or the sender's actual usrp.get_rx_rate()). When set,
    // the rate feeds the FFT bin derivation via the same explicit_span_hz fast-path used offline,
    // and both rate + center are pushed into the CHDR converter so it stamps rx_sample_rate_hz /
    // rx_center_frequency_hz metadata that the detector, visualizer and snipper read.
    auto env_string = [](const char* name) -> std::optional<std::string> {
      const char* value = std::getenv(name);
      if (value != nullptr && value[0] != '\0') {
        return std::string(value);
      }
      return std::nullopt;
    };
    const auto rate_override_str = env_string("USRP_SAMPLE_RATE_HZ");
    const auto center_override_str = env_string("USRP_CENTER_FREQ_HZ");
    std::optional<double> rate_override_hz;
    if (rate_override_str.has_value()) {
      try {
        rate_override_hz = std::stod(*rate_override_str);
        HOLOSCAN_LOG_INFO("USRP_SAMPLE_RATE_HZ override active: {} Hz (supersedes config channel_sample_rates_hz/fft.span)",
                          *rate_override_hz);
      } catch (const std::exception&) {
        HOLOSCAN_LOG_WARN("Ignoring unparseable USRP_SAMPLE_RATE_HZ='{}'", *rate_override_str);
      }
    }
    if (center_override_str.has_value()) {
      HOLOSCAN_LOG_INFO("USRP_CENTER_FREQ_HZ override active: {} Hz", *center_override_str);
    }

    const auto fft_runtime = usrp_wideband::resolve_fft_runtime_config(*this, rate_override_hz);
    const auto fft_span_hz = fft_runtime.active_span_hz;
    const auto fft_span_metadata_hz = static_cast<uint64_t>(std::llround(fft_runtime.active_span_hz));
    if (fft_runtime.num_packets_per_fft > std::numeric_limits<uint16_t>::max()) {
      HOLOSCAN_LOG_ERROR("Derived num_packets_per_fft={} exceeds uint16_t limit for chdr_converter.num_packets_per_fft",
                         fft_runtime.num_packets_per_fft);
      exit(1);
    }
    // Resolve the CHDR rate/center strings: env override wins, else the config value (empty = unset).
    const std::string chdr_sample_rates =
        rate_override_str.value_or(usrp_wideband::from_config_or<std::string>(
            *this, "chdr_converter.channel_sample_rates_hz", std::string{}));
    const std::string chdr_center_freqs =
        center_override_str.value_or(usrp_wideband::from_config_or<std::string>(
            *this, "chdr_converter.channel_center_frequencies_hz", std::string{}));
    auto chdrConverterOp = make_operator<ops::ChdrConverterOpRx>(
      "chdrConverterOp",
      pipeline_shutdown_term,
      from_config("chdr_converter"),
      Arg("num_packets_per_fft") = static_cast<uint16_t>(fft_runtime.num_packets_per_fft),
      Arg("channel_sample_rates_hz") = chdr_sample_rates,
      Arg("channel_center_frequencies_hz") = chdr_center_freqs);
    const int pipeline_channels = std::max(1, from_config("chdr_converter.num_channels").as<int>());

    const bool enable_spectrogram = from_config("pipeline.enable_spectrogram").as<bool>();
    const bool enable_detector = from_config("pipeline.enable_detector").as<bool>();
    // Optional signal-snipper branch: cuts detected signals out of the stream and writes SigMF.
    // Requires the detector (it consumes the detector mask). Defaults off so existing configs are
    // unaffected.
    const bool enable_signal_snipper =
        enable_detector &&
        usrp_wideband::from_config_or<bool>(*this, "pipeline.enable_signal_snipper", false);
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
    const bool coherent_power_fft_aligned_path =
      enable_detector && detector_type == "coherent_power";
    const bool visualization_consumes_fft_directly =
      enable_visualization && coherent_power_fft_aligned_path;
    const bool detector_consumes_fft_directly = coherent_power_fft_aligned_path;
    const bool spectrogram_required =
      enable_spectrogram && (spectrogram_save_enabled ||
                             spectrogram_tensor_save_enabled ||
                             (enable_logger_branch && effective_log_from_spectrogram) ||
                             (enable_detector && !detector_consumes_fft_directly) ||
                             (enable_visualization && !visualization_consumes_fft_directly));
    const bool detector_uses_dino_style_stride =
      enable_detector && detector_type == "cuda_dino";
    const int configured_dino_emit_stride =
      (enable_detector && detector_type == "cuda_dino")
        ? std::max(1, from_config("cuda_dino_detector.emit_stride").as<int>())
        : 1;
    const bool enable_fft_emit_stride =
      bypass_spectrogram_passthrough && detector_uses_dino_style_stride &&
        configured_dino_emit_stride > 1;
    const int visual_render_stride =
      enable_visualization ? std::max(1, from_config("visualization.renderer.render_every_n_frames").as<int>()) : 1;
    const int visual_emit_stride = 1;
    const int visual_mask_emit_stride = 1;

    std::vector<std::shared_ptr<holoscan::Operator>> fftOps;
    fftOps.reserve(static_cast<size_t>(pipeline_channels));
    for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
      fftOps.push_back(make_operator<ops::FFT>(
        std::string("fftOpCh") + std::to_string(channel_index),
        from_config("fft"),
        holoscan::Arg("burst_size") = fft_runtime.actual_fft_size,
        holoscan::Arg("transform_points") = static_cast<uint32_t>(fft_runtime.actual_fft_size),
        holoscan::Arg("window_points") = static_cast<uint32_t>(fft_runtime.actual_fft_size),
        holoscan::Arg("resolution") = fft_runtime.resolution_hz,
        holoscan::Arg("span") = fft_span_metadata_hz,
        holoscan::Arg("f1_index") = fft_runtime.f1_index,
        holoscan::Arg("f2_index") = fft_runtime.f2_index,
        holoscan::Arg("emit_stride") = (enable_fft_emit_stride ? configured_dino_emit_stride : 1)));
    }

    if (enable_detector && !enable_spectrogram) {
      HOLOSCAN_LOG_ERROR("pipeline.enable_detector=true requires pipeline.enable_spectrogram=true");
      exit(1);
    }

    if (enable_detector && detector_type != "cuda_dino" && detector_type != "coherent_power") {
      HOLOSCAN_LOG_ERROR("Unsupported pipeline.detector_type='{}'. Expected 'cuda_dino' or 'coherent_power'.",
                         detector_type);
      exit(1);
    }

    std::vector<std::shared_ptr<holoscan::Operator>> spectrogramOps;
    std::vector<std::shared_ptr<holoscan::Operator>> cudaDinoDetectorOps;
    std::vector<std::shared_ptr<holoscan::Operator>> coherentDetectorOps;
    if (spectrogram_required) {
      spectrogramOps.reserve(static_cast<size_t>(pipeline_channels));
      for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
        spectrogramOps.push_back(make_operator<ops::Spectrogram>(
            std::string("spectrogramOpCh") + std::to_string(channel_index),
            from_config("spectrogram")));
      }
    }
    if (enable_detector) {
      if (detector_type == "cuda_dino") {
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

    std::vector<std::shared_ptr<holoscan::Operator>> signalSnipperOps;
    std::vector<std::shared_ptr<holoscan::Operator>> sigmfFileSinkOps;
    if (enable_signal_snipper) {
      signalSnipperOps.reserve(static_cast<size_t>(pipeline_channels));
      sigmfFileSinkOps.reserve(static_cast<size_t>(pipeline_channels));
      for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
        signalSnipperOps.push_back(make_operator<ops::SignalSnipperOp>(
            std::string("signalSnipperOpCh") + std::to_string(channel_index),
            from_config("signal_snipper"),
            holoscan::Arg("channel_filter") = channel_index));
        sigmfFileSinkOps.push_back(make_operator<ops::SigmfFileSinkOp>(
            std::string("sigmfFileSinkOpCh") + std::to_string(channel_index),
            from_config("sigmf_file_sink")));
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
    if (coherent_power_fft_aligned_path && enable_visualization) {
      HOLOSCAN_LOG_INFO(
          "Routing coherent power detector and visualizer directly from FFT output for exact frame alignment.");
    }

    std::shared_ptr<holoscan::Operator> spectrogramVisualizerOp;
    std::shared_ptr<holoscan::Operator> holovizOp;
    std::vector<std::shared_ptr<holoscan::Operator>> visualSpectrogramGateOps;
    std::vector<std::shared_ptr<holoscan::Operator>> visualMaskGateOps;
    std::vector<std::shared_ptr<holoscan::Operator>> visualSpectrogramStoreOps;
    std::vector<std::shared_ptr<holoscan::Operator>> visualMaskStoreOps;
    if (enable_visualization) {
        auto visualization_shutdown_term = pipeline_shutdown_term;
        g_visualization_shutdown_term = visualization_shutdown_term;
        HOLOSCAN_LOG_INFO("Sharing pipeline_shutdown_term with visualization shutdown handling");
      ops::set_visualization_full_ui_enabled(true);
      const auto tensor_name = from_config("visualization.renderer.tensor_name").as<std::string>();
      const auto center_frequency_hz = from_config("visualization.renderer.center_frequency_hz").as<double>();
      int visualization_refresh_hz = 30;
      try {
        visualization_refresh_hz =
            std::max(1, from_config("visualization.renderer.refresh_hz").as<int>());
      } catch (const std::exception&) {
      }
      const std::string visualization_recess_period = std::to_string(visualization_refresh_hz) + "hz";
        const auto visualization_channel_filter =
          from_config("visualization.renderer.channel_filter").as<int>();
      const std::string detector_label =
          (!enable_detector || detector_type == "cuda_dino")
              ? std::string("Dinov3")
              : std::string("Coherent Power");
      spectrogramVisualizerOp = make_operator<ops::SpectrogramToHolovizOp>(
        "spectrogramVisualizerOp",
        Arg("shutdown_scheduling_term") = visualization_shutdown_term,
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = visualization_recess_period),
        from_config("visualization.renderer"),
        Arg("fft_size") = fft_runtime.actual_fft_size,
        Arg("num_channels") = pipeline_channels,
        Arg("channel_filter") = visualization_channel_filter,
        Arg("center_frequency_hz") = center_frequency_hz,
        Arg("span_hz") = fft_span_hz,
        Arg("detector_label") = detector_label,
        Arg("render_every_n_frames") = visual_render_stride);

      visualSpectrogramGateOps.reserve(static_cast<size_t>(pipeline_channels));
      visualMaskGateOps.reserve(static_cast<size_t>(pipeline_channels));
      visualSpectrogramStoreOps.reserve(static_cast<size_t>(pipeline_channels));
      visualMaskStoreOps.reserve(static_cast<size_t>(pipeline_channels));
      for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
        visualSpectrogramGateOps.push_back(make_operator<ops::SpectrogramPreviewOp>(
            std::string("visualSpectrogramGateOpCh") + std::to_string(channel_index),
          Arg("channel_index") = channel_index,
          Arg("emit_every_n") = visual_emit_stride,
          Arg("output_width") = from_config("visualization.renderer.output_width").as<int>(),
          Arg("output_height") = from_config("visualization.renderer.rows_per_frame").as<int>(),
          Arg("db_floor") = from_config("visualization.renderer.db_floor").as<float>(),
          Arg("db_ceil") = from_config("visualization.renderer.db_ceil").as<float>()));
        visualMaskGateOps.push_back(make_operator<ops::MaskPreviewOp>(
            std::string("visualMaskGateOpCh") + std::to_string(channel_index),
          Arg("channel_index") = channel_index,
          Arg("emit_every_n") = visual_mask_emit_stride,
          Arg("output_width") = from_config("visualization.renderer.output_width").as<int>(),
          Arg("output_height") = from_config("visualization.renderer.rows_per_frame").as<int>()));
        visualSpectrogramStoreOps.push_back(make_operator<ops::SpectrogramPreviewStoreOp>(
          std::string("visualSpectrogramStoreOpCh") + std::to_string(channel_index),
          Arg("allow_backpressure_valve") =
              from_config("visualization.renderer.allow_backpressure_valve").as<bool>()));
        visualMaskStoreOps.push_back(make_operator<ops::MaskPreviewStoreOp>(
          std::string("visualMaskStoreOpCh") + std::to_string(channel_index),
          Arg("allow_backpressure_valve") =
              from_config("visualization.renderer.allow_backpressure_valve").as<bool>()));
      }
        
      holovizOp = make_operator<LoggingHolovizOp>(
        "holovizOp",
        Arg("window_close_scheduling_term") = visualization_shutdown_term,
        Arg("enable_render_buffer_output") = false,
        from_config("visualization.holoviz"),
        Arg("layer_callback",
          ops::HolovizOp::LayerCallbackFunction(
            std::bind(&UsrpWidebandSignalDetectionPipeline::layer_callback,
                  this,
                  std::placeholders::_1))),
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
    if (spectrogram_required) {
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
      }
    }
    if (enable_visualization) {
      for (auto& op : visualSpectrogramGateOps) {
        add_operator(op);
      }
      for (auto& op : visualMaskGateOps) {
        add_operator(op);
      }
      for (auto& op : visualSpectrogramStoreOps) {
        add_operator(op);
      }
      for (auto& op : visualMaskStoreOps) {
        add_operator(op);
      }
      add_operator(spectrogramVisualizerOp);
      add_operator(holovizOp);
    }
    for (auto& op : logOps) {
      add_operator(op);
    }
    if (enable_signal_snipper) {
      for (auto& op : signalSnipperOps) {
        add_operator(op);
      }
      for (auto& op : sigmfFileSinkOps) {
        add_operator(op);
      }
    }
    if (unusedChdrOutputDropOp) {
      add_operator(unusedChdrOutputDropOp);
    }

    for (int channel_index = 0; channel_index < pipeline_channels; ++channel_index) {
      const char* chdr_port = channel_index == 0 ? "out0" : "out1";
      auto& fftOp = fftOps[static_cast<size_t>(channel_index)];
      add_flow(chdrConverterOp, fftOp, {{chdr_port, "in"}});

      std::shared_ptr<holoscan::Operator> detector_source = fftOp;
      std::shared_ptr<holoscan::Operator> logger_source = fftOp;
      if (spectrogram_required) {
        auto& spectrogramOp = spectrogramOps[static_cast<size_t>(channel_index)];
        add_flow(fftOp, spectrogramOp);
        if (!detector_consumes_fft_directly) {
          detector_source = spectrogramOp;
        }
        if (enable_logger_branch && effective_log_from_spectrogram) {
          logger_source = spectrogramOp;
        }

        if (enable_visualization && !visualization_consumes_fft_directly) {
          add_flow(spectrogramOp,
                   visualSpectrogramGateOps[static_cast<size_t>(channel_index)],
                   {{"out", "in"}});
          add_flow(visualSpectrogramGateOps[static_cast<size_t>(channel_index)],
                   visualSpectrogramStoreOps[static_cast<size_t>(channel_index)],
                   {{"out", "in"}});
        }
      }

      if (enable_visualization && visualization_consumes_fft_directly) {
        add_flow(fftOp,
                 visualSpectrogramGateOps[static_cast<size_t>(channel_index)],
                 {{"out", "in"}});
        add_flow(visualSpectrogramGateOps[static_cast<size_t>(channel_index)],
                 visualSpectrogramStoreOps[static_cast<size_t>(channel_index)],
                 {{"out", "in"}});
      }

      if (enable_detector) {
        if (detector_type == "coherent_power") {
          add_flow(detector_source, coherentDetectorOps[static_cast<size_t>(channel_index)]);
        } else if (detector_type == "cuda_dino") {
          add_flow(detector_source, cudaDinoDetectorOps[static_cast<size_t>(channel_index)]);
        }
      }

      if (enable_signal_snipper) {
        auto& snipperOp = signalSnipperOps[static_cast<size_t>(channel_index)];
        // Tap the raw time-domain IQ upstream of the FFT (same port that feeds fftOp).
        add_flow(chdrConverterOp, snipperOp, {{chdr_port, "iq_in"}});
        // Feed the detector mask.
        if (detector_type == "coherent_power") {
          add_flow(coherentDetectorOps[static_cast<size_t>(channel_index)], snipperOp,
                   {{"mask_out", "mask_in"}});
        } else if (detector_type == "cuda_dino") {
          add_flow(cudaDinoDetectorOps[static_cast<size_t>(channel_index)], snipperOp,
                   {{"mask_out", "mask_in"}});
        }
        add_flow(snipperOp, sigmfFileSinkOps[static_cast<size_t>(channel_index)],
                 {{"snippets_out", "in"}});
      }

      if (enable_logger_branch && effective_log_from_spectrogram) {
        if (!enable_spectrogram) {
          HOLOSCAN_LOG_ERROR("pipeline.log_from_spectrogram=true requires pipeline.enable_spectrogram=true");
          exit(1);
        }
        add_flow(logger_source, logOps[static_cast<size_t>(channel_index)]);
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
    if (enable_visualization && enable_detector) {
      for (int ch = 0; ch < pipeline_channels; ++ch) {
        if (detector_type == "coherent_power") {
          add_flow(coherentDetectorOps[static_cast<size_t>(ch)],
                   visualMaskGateOps[static_cast<size_t>(ch)],
                   {{"mask_out", "in"}});
        } else if (detector_type == "cuda_dino") {
          add_flow(cudaDinoDetectorOps[static_cast<size_t>(ch)],
                   visualMaskGateOps[static_cast<size_t>(ch)],
                   {{"mask_out", "in"}});
        } else {
          continue;
        }
        add_flow(visualMaskGateOps[static_cast<size_t>(ch)],
                 visualMaskStoreOps[static_cast<size_t>(ch)],
                 {{"out", "in"}});
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

  sigset_t sigint_set {};
  sigemptyset(&sigint_set);
  sigaddset(&sigint_set, SIGINT);
  if (pthread_sigmask(SIG_BLOCK, &sigint_set, nullptr) != 0) {
    HOLOSCAN_LOG_ERROR("Failed to block SIGINT for dedicated shutdown handling thread");
    return -1;
  }

  const gxf_context_t app_context = app->executor().context();
  std::atomic<bool> signal_thread_exit {false};
  std::thread signal_thread([app_context, sigint_set, &signal_thread_exit]() mutable {
    int interrupt_count = 0;
    while (true) {
      int signal_number = 0;
      const int wait_result = sigwait(&sigint_set, &signal_number);
      if (wait_result != 0) {
        continue;
      }
      if (signal_thread_exit.load(std::memory_order_relaxed)) {
        break;
      }
      if (signal_number != SIGINT) {
        continue;
      }

      interrupt_count += 1;
      std::fprintf(stderr,
                   "[copilot-probe] sigwait received signal=%d count=%d\n",
                   signal_number,
                   interrupt_count);
      std::fflush(stderr);
      HOLOSCAN_LOG_INFO("sigwait received signal={} count={}", signal_number, interrupt_count);
      if (interrupt_count == 1) {
        request_graceful_shutdown(app_context);
        continue;
      }

      std::_Exit(128 + SIGINT);
    }
  });

  app->run();

  signal_thread_exit.store(true, std::memory_order_relaxed);
  pthread_kill(signal_thread.native_handle(), SIGINT);
  signal_thread.join();

  shutdown();
  return 0;
}

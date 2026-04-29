#include "spectrogram_visualization.hpp"

#include <getopt.h>

#include <holoviz/holoviz.hpp>
#include <holoviz/imgui/imgui.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "../../gxf_extensions/yuan_qcap/stb/stb_image_write.h"

namespace {

constexpr const char* kHostMountedSpectrogramDir = "/workspace/spectrograms";
constexpr const char* kHostMountedMaskDir = "/workspace/dino_masks";

struct OfflineReplayOverrides {
  std::string config_path = "old_configs/config_offline_replay.yaml";
  std::string offline_dir;
  std::string mask_dir = kHostMountedMaskDir;
  std::string screenshot_path;
  double frame_rate = -1.0;
  bool repeat = true;
};

std::filesystem::path resolve_config_path(const char* argv0, const std::string& config_arg) {
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

std::string resolve_screenshot_path(const std::string& requested_path) {
  if (requested_path.empty()) {
    return requested_path;
  }

  const std::filesystem::path screenshot_path(requested_path);
  if (screenshot_path.is_absolute()) {
    return screenshot_path.string();
  }

  return (std::filesystem::path(kHostMountedSpectrogramDir) / screenshot_path).string();
}

int export_first_frame_png(const std::string& directory,
                           const std::string& mask_directory,
                           const std::string& output_path) {
  const auto frames = holoscan::ops::list_offline_pgm_frames(directory, -1);
  if (frames.empty()) {
    HOLOSCAN_LOG_ERROR("No .pgm spectrogram frames found in '{}'", directory);
    return -1;
  }

  constexpr int kNumChannels = 2;
  constexpr int kHistoryFrames = 5;
  constexpr float kBlueLimit = 0.10f;
  constexpr float kRedLimit = 0.92f;
  std::vector<holoscan::ops::ChannelVisualizationState> channel_states(static_cast<size_t>(kNumChannels));
  for (const auto& frame_path : frames) {
    holoscan::ops::OfflinePgmFrame frame;
    if (!holoscan::ops::load_offline_pgm_frame(frame_path, frame)) {
      HOLOSCAN_LOG_WARN("Skipping unreadable spectrogram frame '{}'", frame_path.string());
      continue;
    }

    int channel = -1;
    uint64_t frame_number = 0;
    int rows = 0;
    int cols = 0;
    if (!holoscan::ops::parse_recorded_pgm_name(frame_path.filename().string(),
                                                "spectrogram",
                                                channel,
                                                frame_number,
                                                rows,
                                                cols)) {
      continue;
    }
    if (channel < 0 || channel >= kNumChannels) {
      continue;
    }

    auto& state = channel_states[static_cast<size_t>(channel)];
    holoscan::ops::append_spectrogram_history(state, frame.pixels, frame.width, frame.height, kHistoryFrames);
    state.current_psd_trace = holoscan::ops::compute_psd_trace(frame.pixels, frame.width, frame.height);
    holoscan::ops::update_max_hold_trace(state.current_psd_trace, state.max_hold_trace);

    holoscan::ops::OfflinePgmFrame mask_frame;
    const auto mask_path = holoscan::ops::find_matching_recorded_pgm(mask_directory,
                                                                     "dino_mask",
                                                                     channel,
                                                                     frame_number);
    const bool has_mask = !mask_path.empty() && holoscan::ops::load_offline_pgm_frame(mask_path, mask_frame);
    const auto density_trace = has_mask ? holoscan::ops::compute_density_trace(&mask_frame)
                                        : holoscan::ops::compute_density_trace_from_grayscale(frame.pixels,
                                                                                              frame.width,
                                                                                              frame.height,
                                                                                              kRedLimit);
    holoscan::ops::update_density_history(density_trace, state.density_trace, state.density_frames_seen);
    state.latest_mask = has_mask ? mask_frame : holoscan::ops::OfflinePgmFrame{};
    state.overlay_available = has_mask;
    state.active = true;
    state.info.channel = channel;
    state.info.frame_number = static_cast<int64_t>(frame_number);
    state.info.center_frequency_hz = 0.0;
    state.info.fft_size = 20480;
    state.info.dino_chunk_rows = frame.height;
    state.info.dino_chunk_cols = frame.width;
    state.info.overlay_available = has_mask;
    state.info.title = "USRP WIDEBAND";
    state.info.subtitle = "OFFLINE REPLAY";
  }

  int composed_width = 0;
  int composed_height = 0;
  int panel_width = 512;
  int panel_height = 256;
  for (const auto& state : channel_states) {
    if (!state.active) {
      continue;
    }
    panel_width = std::max(panel_width, state.history_width);
    panel_height = std::max(panel_height,
                            std::max(1,
                                     state.history_capacity_rows > 0 ? state.history_capacity_rows
                                                                     : state.history_valid_rows));
  }
  auto composed = holoscan::ops::compose_visualization_rgb(channel_states,
                                                           kBlueLimit,
                                                           kRedLimit,
                                                           0.38f,
                                                           true,
                                                           panel_width,
                                                           panel_height,
                                                           composed_width,
                                                           composed_height);

  const std::filesystem::path png_path(output_path);
  if (!png_path.parent_path().empty()) {
    std::filesystem::create_directories(png_path.parent_path());
  }

  const int stride = composed_width * 3;
  if (!stbi_write_png(png_path.string().c_str(), composed_width, composed_height, 3, composed.data(), stride)) {
    throw std::runtime_error("Failed to write PNG screenshot to " + png_path.string());
  }

  HOLOSCAN_LOG_INFO("Saved offline analyzer screenshot '{}' from {} replayed frames", png_path.string(), frames.size());
  return 0;
}

void usage(const char* argv0) {
  HOLOSCAN_LOG_INFO("Usage: {} [--config FILE] [--offline-dir DIR] [--mask-dir DIR] [--fps FPS] [--no-loop] [--screenshot FILE.png]", argv0);
}

OfflineReplayOverrides parse_arguments(int argc, char** argv) {
  OfflineReplayOverrides options;

  static option long_options[] = {{"config", required_argument, nullptr, 'c'},
                                  {"offline-dir", required_argument, nullptr, 'd'},
                                  {"mask-dir", required_argument, nullptr, 'm'},
                                  {"fps", required_argument, nullptr, 'f'},
                                  {"screenshot", required_argument, nullptr, 's'},
                                  {"no-loop", no_argument, nullptr, 'n'},
                                  {"help", no_argument, nullptr, 'h'},
                                  {0, 0, 0, 0}};

  while (true) {
    const int opt = getopt_long(argc, argv, "c:d:m:f:s:nh", long_options, nullptr);
    if (opt == -1) {
      break;
    }

    switch (opt) {
      case 'c':
        options.config_path = optarg;
        break;
      case 'd':
        options.offline_dir = optarg;
        break;
      case 'm':
        options.mask_dir = optarg;
        break;
      case 'f':
        options.frame_rate = std::stod(optarg);
        break;
      case 's':
        options.screenshot_path = optarg;
        break;
      case 'n':
        options.repeat = false;
        break;
      case 'h':
        usage(argv[0]);
        std::exit(0);
      default:
        usage(argv[0]);
        std::exit(1);
    }
  }

  return options;
}

}  // namespace

class OfflineSpectrogramReplayApp : public holoscan::Application {
 public:
  void set_overrides(OfflineReplayOverrides overrides) {
    overrides_ = std::move(overrides);
  }

  void layer_callback(const std::vector<holoscan::gxf::Entity>&) {
    holoscan::viz::BeginImGuiLayer();
    holoscan::ops::render_visualization_ui_overlay();
    holoscan::viz::EndLayer();
  }

  void compose() override {
    using namespace holoscan;

    ops::set_visualization_full_ui_enabled(true);

    const auto tensor_name = from_config("offline_replay.tensor_name").as<std::string>();
    const auto directory = overrides_.offline_dir.empty() ? from_config("offline_replay.directory").as<std::string>()
                                                          : overrides_.offline_dir;
    const auto frame_rate = overrides_.frame_rate > 0.0 ? overrides_.frame_rate
                                                        : from_config("offline_replay.frame_rate").as<double>();
    const auto fft_span_hz = from_config("fft.span").as<double>();
    const auto center_frequency_hz = from_config("visualization.renderer.center_frequency_hz").as<double>();
    const auto detector_type = from_config("pipeline.detector_type").as<std::string>();
    const std::string detector_label = detector_type == "coherent_power" ? "Coherent Power" : "Dinov3";

    auto replay = make_operator<ops::OfflinePgmReplayOp>("offlineReplayOp",
                                                         from_config("offline_replay"),
                                                         make_condition<BooleanCondition>("replay_active", true),
                                                         Arg("directory") = directory,
                                                         Arg("mask_directory") = overrides_.mask_dir,
                                                         Arg("repeat") = overrides_.repeat,
                               Arg("frame_rate") = frame_rate,
                               Arg("center_frequency_hz") = center_frequency_hz,
                               Arg("span_hz") = fft_span_hz,
                               Arg("detector_label") = detector_label);

    auto holoviz = make_operator<ops::HolovizOp>("holoviz",
                                                 from_config("visualization.holoviz"),
                           Arg("layer_callback",
                             ops::HolovizOp::LayerCallbackFunction(
                               std::bind(&OfflineSpectrogramReplayApp::layer_callback,
                                     this,
                                     std::placeholders::_1))),
                                                 Arg("headless") = false,
                                                 Arg("enable_render_buffer_output") = false,
                                                 Arg("tensors") = ops::make_spectrogram_input_specs(tensor_name));

    add_flow(replay, holoviz, {{"outputs", "receivers"}});
  }

 private:
  OfflineReplayOverrides overrides_;
};

int main(int argc, char** argv) {
  auto overrides = parse_arguments(argc, argv);
  overrides.screenshot_path = resolve_screenshot_path(overrides.screenshot_path);

  if (!overrides.screenshot_path.empty()) {
    const auto directory = overrides.offline_dir.empty() ? kHostMountedSpectrogramDir : overrides.offline_dir;
    return export_first_frame_png(directory, overrides.mask_dir, overrides.screenshot_path);
  }

  auto app = holoscan::make_application<OfflineSpectrogramReplayApp>();
  app->set_overrides(overrides);

  const auto config_path = resolve_config_path(argv[0], overrides.config_path);
  if (!std::filesystem::exists(config_path)) {
    HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist", static_cast<std::string>(config_path));
    return -1;
  }

  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("event-based-scheduler",
                                                                    app->from_config("scheduler")));
  app->run();
  return 0;
}
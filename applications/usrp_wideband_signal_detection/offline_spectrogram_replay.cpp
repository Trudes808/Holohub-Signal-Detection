#include "spectrogram_visualization.hpp"

#include <getopt.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "../../gxf_extensions/yuan_qcap/stb/stb_image_write.h"

namespace {

constexpr const char* kHostMountedSpectrogramDir = "/workspace/spectrograms";

struct OfflineReplayOverrides {
  std::string config_path = "config_offline_replay.yaml";
  std::string offline_dir;
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

int export_first_frame_png(const std::string& directory, const std::string& output_path) {
  const auto frames = holoscan::ops::list_offline_pgm_frames(directory, -1);
  if (frames.empty()) {
    HOLOSCAN_LOG_ERROR("No .pgm spectrogram frames found in '{}'", directory);
    return -1;
  }

  holoscan::ops::OfflinePgmFrame frame;
  if (!holoscan::ops::load_offline_pgm_frame(frames.front(), frame)) {
    HOLOSCAN_LOG_ERROR("Failed to load spectrogram frame '{}'", frames.front().string());
    return -1;
  }

  auto rgb = holoscan::ops::colorize_grayscale_spectrogram(frame.pixels);

  const std::filesystem::path png_path(output_path);
  if (!png_path.parent_path().empty()) {
    std::filesystem::create_directories(png_path.parent_path());
  }

  const int stride = frame.width * 3;
  if (!stbi_write_png(png_path.string().c_str(), frame.width, frame.height, 3, rgb.data(), stride)) {
    throw std::runtime_error("Failed to write PNG screenshot to " + png_path.string());
  }

  HOLOSCAN_LOG_INFO("Saved offline spectrogram screenshot '{}' from '{}'", png_path.string(), frames.front().string());
  return 0;
}

void usage(const char* argv0) {
  HOLOSCAN_LOG_INFO("Usage: {} [--config FILE] [--offline-dir DIR] [--fps FPS] [--no-loop] [--screenshot FILE.png]", argv0);
}

OfflineReplayOverrides parse_arguments(int argc, char** argv) {
  OfflineReplayOverrides options;

  static option long_options[] = {{"config", required_argument, nullptr, 'c'},
                                  {"offline-dir", required_argument, nullptr, 'd'},
                                  {"fps", required_argument, nullptr, 'f'},
                                  {"screenshot", required_argument, nullptr, 's'},
                                  {"no-loop", no_argument, nullptr, 'n'},
                                  {"help", no_argument, nullptr, 'h'},
                                  {0, 0, 0, 0}};

  while (true) {
    const int opt = getopt_long(argc, argv, "c:d:f:s:nh", long_options, nullptr);
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

  void compose() override {
    using namespace holoscan;

    const auto tensor_name = from_config("offline_replay.tensor_name").as<std::string>();
    const auto directory = overrides_.offline_dir.empty() ? from_config("offline_replay.directory").as<std::string>()
                                                          : overrides_.offline_dir;
    const auto frame_rate = overrides_.frame_rate > 0.0 ? overrides_.frame_rate
                                                        : from_config("offline_replay.frame_rate").as<double>();

    auto replay = make_operator<ops::OfflinePgmReplayOp>("offlineReplayOp",
                                                         from_config("offline_replay"),
                                                         make_condition<BooleanCondition>("replay_active", true),
                                                         Arg("directory") = directory,
                                                         Arg("repeat") = overrides_.repeat,
                                                         Arg("frame_rate") = frame_rate);

    auto holoviz = make_operator<ops::HolovizOp>("holoviz",
                                                 from_config("visualization.holoviz"),
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
    return export_first_frame_png(directory, overrides.screenshot_path);
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
// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>
#include <imgui.h>

#include <getopt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace holoscan::ops {

class OfflineSpectrogramSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OfflineSpectrogramSourceOp)

  void setup(OperatorSpec& spec) override {
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(directory_,
               "directory",
               "Directory",
               "Directory containing spectrogram .pgm files.",
               std::string("/tmp/usrp_spectrograms"));
    spec.param(loop_, "loop", "Loop", "Loop back to the first frame after the last frame.", true);
    spec.param(target_fps_,
               "target_fps",
               "Target FPS",
               "Frame replay rate for offline playback. Set to 0 for no delay.",
               8.0);
    spec.param(color_mode_,
           "color_mode",
           "Color mode",
           "Color mode: gray or heatmap.",
           std::string("heatmap"));
  }

  void initialize() override {
    load_frames();
    Operator::initialize();
  }

  void compute(InputContext&, OutputContext& output, ExecutionContext& context) override {
    if (frames_.empty()) {
      throw std::runtime_error("Offline spectrogram source has no frames to emit.");
    }

    apply_frame_pacing();

    if (frame_index_ >= frames_.size()) {
      if (!loop_.get()) {
        frame_index_ = frames_.size() - 1;
      } else {
        frame_index_ = 0;
      }
    }

    const Frame& frame = frames_[frame_index_];

    auto entity = holoscan::gxf::Entity::New(&context);
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>("image");
    if (!tensor) {
      throw std::runtime_error("Failed to allocate output tensor 'image'.");
    }

    const nvidia::gxf::Shape shape{frame.height, frame.width, 3};
    const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    const uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
    const auto strides = nvidia::gxf::ComputeTrivialStrides(shape, element_size);

    tensor.value()->wrapMemory(shape,
                               element_type,
                               element_size,
                               strides,
                               nvidia::gxf::MemoryStorageType::kSystem,
                               const_cast<uint8_t*>(frame.rgb.data()),
                               nullptr);

    auto meta = metadata();
    meta->set("offline_frame_index", static_cast<int64_t>(frame_index_));
    meta->set("offline_frame_name", frame.path.filename().string());
    meta->set("offline_frame_width", frame.width);
    meta->set("offline_frame_height", frame.height);

    output.emit(entity, "output");

    ++frame_index_;
  }

 private:
  struct Frame {
    fs::path path;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
  };

  static std::optional<std::string> read_token(std::istream& input) {
    std::string token;
    char ch = '\0';

    while (input.get(ch)) {
      if (std::isspace(static_cast<unsigned char>(ch))) {
        continue;
      }
      if (ch == '#') {
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        continue;
      }
      token.push_back(ch);
      break;
    }

    if (token.empty()) {
      return std::nullopt;
    }

    while (input.get(ch)) {
      if (std::isspace(static_cast<unsigned char>(ch))) {
        break;
      }
      if (ch == '#') {
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        break;
      }
      token.push_back(ch);
    }

    return token;
  }

  static Frame load_pgm_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open spectrogram file: " + path.string());
    }

    const auto magic = read_token(input);
    if (!magic || *magic != "P5") {
      throw std::runtime_error("Unsupported PGM format in file: " + path.string());
    }

    const auto width_token = read_token(input);
    const auto height_token = read_token(input);
    const auto max_value_token = read_token(input);
    if (!width_token || !height_token || !max_value_token) {
      throw std::runtime_error("Invalid PGM header in file: " + path.string());
    }

    Frame frame;
    frame.path = path;
    frame.width = std::stoi(*width_token);
    frame.height = std::stoi(*height_token);
    const int max_value = std::stoi(*max_value_token);

    if (frame.width <= 0 || frame.height <= 0) {
      throw std::runtime_error("Invalid PGM dimensions in file: " + path.string());
    }
    if (max_value != 255) {
      throw std::runtime_error("Only 8-bit PGM files are supported: " + path.string());
    }

    const size_t pixel_count = static_cast<size_t>(frame.width) * static_cast<size_t>(frame.height);
    std::vector<uint8_t> gray(pixel_count);
    input.read(reinterpret_cast<char*>(gray.data()), static_cast<std::streamsize>(gray.size()));
    if (input.gcount() != static_cast<std::streamsize>(gray.size())) {
      throw std::runtime_error("Failed to read complete PGM payload: " + path.string());
    }

    frame.rgb.resize(pixel_count * 3);
    for (size_t index = 0; index < pixel_count; ++index) {
      const uint8_t value = gray[index];
      const auto rgb = colorize(value);
      frame.rgb[index * 3 + 0] = rgb[0];
      frame.rgb[index * 3 + 1] = rgb[1];
      frame.rgb[index * 3 + 2] = rgb[2];
    }

    return frame;
  }

  std::array<uint8_t, 3> colorize(uint8_t gray_value) const {
    if (color_mode_.get() == "gray") {
      return {gray_value, gray_value, gray_value};
    }

    constexpr std::array<std::array<float, 3>, 6> kHeatmap = {{
        {0.02f, 0.04f, 0.18f},
        {0.16f, 0.11f, 0.42f},
        {0.38f, 0.14f, 0.55f},
        {0.73f, 0.24f, 0.33f},
        {0.96f, 0.58f, 0.12f},
        {0.99f, 0.92f, 0.39f},
    }};

    const float t = static_cast<float>(gray_value) / 255.0f;
    const float scaled = t * static_cast<float>(kHeatmap.size() - 1);
    const size_t idx0 = static_cast<size_t>(std::clamp<int>(static_cast<int>(scaled), 0, static_cast<int>(kHeatmap.size() - 1)));
    const size_t idx1 = std::min(idx0 + 1, kHeatmap.size() - 1);
    const float alpha = scaled - static_cast<float>(idx0);

    std::array<uint8_t, 3> rgb{};
    for (size_t channel = 0; channel < 3; ++channel) {
      const float value = (1.0f - alpha) * kHeatmap[idx0][channel] + alpha * kHeatmap[idx1][channel];
      rgb[channel] = static_cast<uint8_t>(std::clamp(value * 255.0f, 0.0f, 255.0f));
    }
    return rgb;
  }

  void load_frames() {
    const fs::path directory_path(directory_.get());
    if (!fs::exists(directory_path)) {
      throw std::runtime_error("Offline spectrogram directory does not exist: " + directory_path.string());
    }
    if (!fs::is_directory(directory_path)) {
      throw std::runtime_error("Offline spectrogram path is not a directory: " + directory_path.string());
    }

    std::vector<fs::path> frame_paths;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().extension() == ".pgm") {
        frame_paths.push_back(entry.path());
      }
    }

    std::sort(frame_paths.begin(), frame_paths.end());
    if (frame_paths.empty()) {
      throw std::runtime_error("No .pgm spectrogram files found in: " + directory_path.string());
    }

    frames_.clear();
    frames_.reserve(frame_paths.size());
    for (const auto& path : frame_paths) {
      frames_.push_back(load_pgm_file(path));
    }

    HOLOSCAN_LOG_INFO("Loaded {} offline spectrogram frames from {}",
                      frames_.size(),
                      directory_path.string());
  }

  void apply_frame_pacing() {
    const double fps = target_fps_.get();
    if (fps <= 0.0) {
      return;
    }

    const auto frame_interval = std::chrono::duration<double>(1.0 / fps);
    const auto now = std::chrono::steady_clock::now();
    if (last_emit_time_) {
      const auto next_emit_time = *last_emit_time_ + frame_interval;
      if (now < next_emit_time) {
        std::this_thread::sleep_until(next_emit_time);
      }
    }
    last_emit_time_ = std::chrono::steady_clock::now();
  }

  Parameter<std::string> directory_;
  Parameter<bool> loop_;
  Parameter<double> target_fps_;
  Parameter<std::string> color_mode_;
  std::vector<Frame> frames_;
  size_t frame_index_ = 0;
  std::optional<std::chrono::steady_clock::time_point> last_emit_time_;
};

}  // namespace holoscan::ops

namespace {

struct ViewerOptions {
  std::string offline_dir = "/tmp/usrp_spectrograms";
  double fps = 8.0;
  bool loop = true;
  int count = -1;
  std::string window_title = "USRP Spectrogram Viewer";
  std::string color_mode = "heatmap";
  bool show_fake_overlay = true;
};

class OfflineSpectrogramViewerApp : public holoscan::Application {
 public:
  explicit OfflineSpectrogramViewerApp(ViewerOptions options) : options_(std::move(options)) {}

  void compose() override {
    using namespace holoscan;

    auto source_args = std::vector<Arg>{Arg("directory", options_.offline_dir),
                                        Arg("loop", options_.loop),
                                        Arg("target_fps", options_.fps),
                                        Arg("color_mode", options_.color_mode)};

    std::shared_ptr<holoscan::ops::OfflineSpectrogramSourceOp> source;
    if (options_.count > 0) {
      source = make_operator<holoscan::ops::OfflineSpectrogramSourceOp>(
          "offline_source",
          make_condition<CountCondition>("count-condition", options_.count),
          source_args[0],
          source_args[1],
          source_args[2],
          source_args[3]);
    } else {
      source = make_operator<holoscan::ops::OfflineSpectrogramSourceOp>(
          "offline_source", source_args[0], source_args[1], source_args[2], source_args[3]);
    }

    auto input_spec = ops::HolovizOp::InputSpec("image", ops::HolovizOp::InputType::COLOR);
    auto viewer = make_operator<ops::HolovizOp>(
        "viewer",
        Arg("window_title", options_.window_title),
        Arg("layer_callback",
            ops::HolovizOp::LayerCallbackFunction(
                std::bind(&OfflineSpectrogramViewerApp::layer_callback, this, std::placeholders::_1))),
        Arg("tensors", std::vector<ops::HolovizOp::InputSpec>{input_spec}));

    add_flow(source, viewer, {{"output", "receivers"}});
  }

  void layer_callback(const std::vector<holoscan::gxf::Entity>&) {
    using namespace holoscan;

    viz::BeginImGuiLayer();
    ImGui::Begin("Viewer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("Offline spectrogram replay");
    ImGui::Text("Directory: %s", options_.offline_dir.c_str());
    ImGui::Checkbox("Show fake detections", &options_.show_fake_overlay);
    ImGui::Text("Color mode: %s", options_.color_mode.c_str());
    ImGui::Text("Replay FPS: %.1f", options_.fps);
    ImGui::End();
    viz::EndLayer();

    if (!options_.show_fake_overlay) {
      return;
    }

    viz::BeginGeometryLayer();
    viz::Color(0.10f, 0.95f, 0.80f, 1.0f);
    viz::Text(0.18f, 0.30f, 0.045f, "SIG A 0.91");
    viz::Text(0.62f, 0.58f, 0.045f, "SIG B 0.84");
    viz::Color(1.0f, 0.75f, 0.10f, 1.0f);
    viz::Text(0.12f, 0.86f, 0.032f, "offline overlay prototype");
    viz::EndLayer();
  }

 private:
  ViewerOptions options_;
};

ViewerOptions parse_arguments(int argc, char** argv) {
  ViewerOptions options;

  const option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                 {"offline-dir", required_argument, nullptr, 'd'},
                                 {"fps", required_argument, nullptr, 'f'},
                                 {"count", required_argument, nullptr, 'c'},
                                 {"window-title", required_argument, nullptr, 'w'},
                                 {"no-loop", no_argument, nullptr, 'n'},
                                 {"color-mode", required_argument, nullptr, 'm'},
                                 {"hide-fake-overlay", no_argument, nullptr, 'o'},
                                 {nullptr, 0, nullptr, 0}};

  while (true) {
    int option_index = 0;
    const int opt = getopt_long(argc, argv, "hd:f:c:w:nm:o", long_options, &option_index);
    if (opt == -1) {
      break;
    }

    switch (opt) {
      case 'h':
        std::cout << "Offline spectrogram visualizer\n"
                  << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -h, --help                  Show this help message\n"
                  << "  -d, --offline-dir <DIR>     Directory of .pgm spectrogram files\n"
                  << "  -f, --fps <FPS>             Replay rate in frames per second\n"
                  << "  -c, --count <COUNT>         Stop after COUNT frames\n"
                  << "  -w, --window-title <TITLE>  Window title override\n"
                  << "  -n, --no-loop               Do not loop after the last frame\n"
                  << "  -m, --color-mode <MODE>     gray or heatmap\n"
                  << "  -o, --hide-fake-overlay     Start without fake overlay labels\n";
        std::exit(0);
      case 'd':
        options.offline_dir = optarg;
        break;
      case 'f':
        options.fps = std::stod(optarg);
        break;
      case 'c':
        options.count = std::stoi(optarg);
        break;
      case 'w':
        options.window_title = optarg;
        break;
      case 'n':
        options.loop = false;
        break;
      case 'm':
        options.color_mode = optarg;
        break;
      case 'o':
        options.show_fake_overlay = false;
        break;
      case '?':
        break;
      default:
        throw std::runtime_error("Unhandled command line option.");
    }
  }

  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto options = parse_arguments(argc, argv);

    if (options.color_mode != "gray" && options.color_mode != "heatmap") {
      throw std::runtime_error("--color-mode must be 'gray' or 'heatmap'");
    }

    auto app = holoscan::make_application<OfflineSpectrogramViewerApp>(options);
    app->run();
    return 0;
  } catch (const std::exception& exception) {
    HOLOSCAN_LOG_ERROR("Offline spectrogram visualizer failed: {}", exception.what());
    return -1;
  }
}
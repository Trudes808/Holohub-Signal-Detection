// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda_dino_detector.hpp>
#include <cuda_dino_types.hpp>
#include <holoscan/holoscan.hpp>

#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <getopt.h>

namespace {

using ReplayTensor = holoscan::ops::cuda_dino_in_t;
using ReplayComplex = holoscan::ops::cuda_dino_complex;

struct ReplayOverrides {
  std::string config_path = "config_cuda_dino_scaffold_single_channel.yaml";
  std::string tensor_path;
  std::string output_dir;
  std::string tensor_axis_order = "auto";
  double span_hz = -1.0;
  int channel_number = 0;
  int debug_chunk_index = 13;
};

struct NpyArray2D {
  std::string descr;
  int rows = 0;
  int cols = 0;
  std::vector<uint8_t> payload;
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

NpyArray2D load_npy_2d(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open npy file: " + path.string());
  }

  char magic[6] {};
  in.read(magic, 6);
  if (std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    throw std::runtime_error("unsupported npy magic in: " + path.string());
  }

  unsigned char version[2] {};
  in.read(reinterpret_cast<char*>(version), 2);
  uint32_t header_len = 0;
  if (version[0] == 1) {
    unsigned char bytes[2] {};
    in.read(reinterpret_cast<char*>(bytes), 2);
    header_len = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U);
  } else {
    unsigned char bytes[4] {};
    in.read(reinterpret_cast<char*>(bytes), 4);
    header_len = static_cast<uint32_t>(bytes[0]) |
                 (static_cast<uint32_t>(bytes[1]) << 8U) |
                 (static_cast<uint32_t>(bytes[2]) << 16U) |
                 (static_cast<uint32_t>(bytes[3]) << 24U);
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header.size()));

  std::smatch descr_match;
  const std::regex descr_pattern("'descr'\\s*:\\s*'([^']+)'");
  if (!std::regex_search(header, descr_match, descr_pattern)) {
    throw std::runtime_error("npy header missing descr in: " + path.string());
  }

  std::smatch shape_match;
  const std::regex shape_pattern("'shape'\\s*:\\s*\\((\\d+)\\s*,\\s*(\\d+)\\s*\\)");
  if (!std::regex_search(header, shape_match, shape_pattern)) {
    throw std::runtime_error("npy header missing 2D shape in: " + path.string());
  }

  NpyArray2D array;
  array.descr = descr_match[1].str();
  array.rows = std::stoi(shape_match[1].str());
  array.cols = std::stoi(shape_match[2].str());
  const size_t payload_bytes = static_cast<size_t>(array.rows) * static_cast<size_t>(array.cols) *
                               (array.descr == "<c8" ? sizeof(float) * 2 : sizeof(float));
  array.payload.resize(payload_bytes);
  in.read(reinterpret_cast<char*>(array.payload.data()), static_cast<std::streamsize>(array.payload.size()));
  if (!in) {
    throw std::runtime_error("truncated npy payload in: " + path.string());
  }
  return array;
}

std::vector<ReplayComplex> to_complex_tensor(const NpyArray2D& array) {
  if (array.descr != "<c8") {
    throw std::runtime_error("expected complex64 npy tensor");
  }
  std::vector<ReplayComplex> output(static_cast<size_t>(array.rows) * static_cast<size_t>(array.cols));
  for (size_t index = 0; index < output.size(); ++index) {
    float components[2] {};
    std::memcpy(components, array.payload.data() + index * sizeof(float) * 2, sizeof(components));
    output[index] = ReplayComplex(components[0], components[1]);
  }
  return output;
}

std::vector<ReplayComplex> transpose_complex_tensor(const std::vector<ReplayComplex>& input,
                                                    int rows,
                                                    int cols) {
  std::vector<ReplayComplex> output(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      output[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] =
          input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    }
  }
  return output;
}

std::string normalize_axis_order(std::string axis_order, int rows, int cols) {
  for (char& ch : axis_order) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (axis_order.empty() || axis_order == "auto") {
    return rows < cols ? "time_frequency" : "frequency_time";
  }
  if (axis_order == "time_frequency" || axis_order == "frequency_time") {
    return axis_order;
  }
  throw std::runtime_error("unsupported tensor_axis_order: " + axis_order);
}

void usage(const char* argv0) {
  HOLOSCAN_LOG_INFO("Usage: {} --tensor-npy FILE [--output-dir DIR] [--config FILE] [--span-hz HZ] [--channel N] [--debug-chunk-index N] [--tensor-axis-order auto|time_frequency|frequency_time]", argv0);
}

ReplayOverrides parse_arguments(int argc, char** argv) {
  ReplayOverrides options;
  static option long_options[] = {{"config", required_argument, nullptr, 'c'},
                                  {"tensor-npy", required_argument, nullptr, 't'},
                                  {"output-dir", required_argument, nullptr, 'o'},
                                  {"span-hz", required_argument, nullptr, 's'},
                                  {"channel", required_argument, nullptr, 'n'},
                                  {"debug-chunk-index", required_argument, nullptr, 'd'},
                                  {"tensor-axis-order", required_argument, nullptr, 'a'},
                                  {"help", no_argument, nullptr, 'h'},
                                  {0, 0, 0, 0}};

  while (true) {
    const int opt = getopt_long(argc, argv, "c:t:o:s:n:d:a:h", long_options, nullptr);
    if (opt == -1) {
      break;
    }
    switch (opt) {
      case 'c':
        options.config_path = optarg;
        break;
      case 't':
        options.tensor_path = optarg;
        break;
      case 'o':
        options.output_dir = optarg;
        break;
      case 's':
        options.span_hz = std::stod(optarg);
        break;
      case 'n':
        options.channel_number = std::stoi(optarg);
        break;
      case 'd':
        options.debug_chunk_index = std::stoi(optarg);
        break;
      case 'a':
        options.tensor_axis_order = optarg;
        break;
      case 'h':
        usage(argv[0]);
        std::exit(0);
      default:
        usage(argv[0]);
        std::exit(1);
    }
  }

  if (options.tensor_path.empty()) {
    usage(argv[0]);
    std::exit(1);
  }

  return options;
}

class OfflineCudaDinoTensorReplayOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OfflineCudaDinoTensorReplayOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<ReplayTensor>("out");
    spec.param(tensor_path_, "tensor_path", "Tensor Path", "Offline complex64 tensor snapshot to replay.");
    spec.param(span_hz_, "span_hz", "Span Hz", "Sample-rate or FFT span used to set detector metadata.", 0.0);
    spec.param(channel_number_, "channel_number", "Channel Number", "Channel number metadata for replayed frame.", 0);
    spec.param(tensor_axis_order_, "tensor_axis_order", "Tensor Axis Order", "Input tensor axis order: auto, time_frequency, or frequency_time.", std::string("auto"));
  }

  void initialize() override {
    Operator::initialize();

    const auto tensor_path = std::filesystem::path(tensor_path_.get());
    const auto array = load_npy_2d(tensor_path);
    auto host_tensor = to_complex_tensor(array);
    auto axis_order = normalize_axis_order(tensor_axis_order_.get(), array.rows, array.cols);

    if (axis_order == "time_frequency") {
      host_tensor = transpose_complex_tensor(host_tensor, array.rows, array.cols);
      rows_ = array.cols;
      cols_ = array.rows;
    } else {
      rows_ = array.rows;
      cols_ = array.cols;
    }

    host_tensor_ = std::move(host_tensor);
    make_tensor(device_tensor_, {static_cast<matx::index_t>(rows_), static_cast<matx::index_t>(cols_)});

    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("failed to create replay CUDA stream");
    }

    const size_t bytes = static_cast<size_t>(rows_) * static_cast<size_t>(cols_) * sizeof(ReplayComplex);
    if (cudaMemcpy(device_tensor_.Data(), host_tensor_.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("failed to upload replay tensor to device");
    }
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output, holoscan::ExecutionContext&) override {
    auto meta = metadata();
    const double span_hz = span_hz_.get();
    if (meta) {
      meta->set("channel_number", static_cast<uint16_t>(std::max(0, channel_number_.get())));
      if (span_hz > 0.0) {
        meta->set("sample_rate_hz", span_hz);
        meta->set("span", static_cast<uint64_t>(std::llround(span_hz)));
        meta->set("resolution", static_cast<uint64_t>(std::llround(span_hz / static_cast<double>(std::max(1, rows_)))));
      }
      meta->set("fft_emit_stride", 1);
      meta->set("fft_emitted_frame_number", static_cast<uint64_t>(1));
    }

    op_output.emit(ReplayTensor{device_tensor_, stream_}, "out");
  }

  void stop() override {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    Operator::stop();
  }

 private:
  holoscan::Parameter<std::string> tensor_path_;
  holoscan::Parameter<double> span_hz_;
  holoscan::Parameter<int> channel_number_;
  holoscan::Parameter<std::string> tensor_axis_order_;

  matx::tensor_t<ReplayComplex, 2> device_tensor_;
  std::vector<ReplayComplex> host_tensor_;
  cudaStream_t stream_ = nullptr;
  int rows_ = 0;
  int cols_ = 0;
};

class OfflineCudaDinoOperatorReplayApp : public holoscan::Application {
 public:
  void set_overrides(ReplayOverrides overrides) {
    overrides_ = std::move(overrides);
  }

  void compose() override {
    using namespace holoscan;

    const double span_hz = overrides_.span_hz > 0.0 ? overrides_.span_hz : from_config("fft.span").as<double>();
    auto replay = make_operator<OfflineCudaDinoTensorReplayOp>(
        "offlineCudaDinoTensorReplayOp",
        make_condition<CountCondition>("replay_once", 1),
        Arg("tensor_path") = overrides_.tensor_path,
        Arg("span_hz") = span_hz,
        Arg("channel_number") = overrides_.channel_number,
        Arg("tensor_axis_order") = overrides_.tensor_axis_order);

    auto detector = make_operator<holoscan::ops::CudaDinoDetector>(
        "cudaDinoDetectorOpCh0",
        from_config("cuda_dino_detector"),
        Arg("channel_filter") = overrides_.channel_number,
        Arg("emit_stride") = 1,
        Arg("debug_mode") = true,
        Arg("enable_debug_artifact_host_copy") = true,
        Arg("debug_chunk_index") = overrides_.debug_chunk_index,
        Arg("debug_artifact_output_dir") = overrides_.output_dir);

    add_flow(replay, detector);
  }

 private:
  ReplayOverrides overrides_;
};

}  // namespace

int main(int argc, char** argv) {
  auto overrides = parse_arguments(argc, argv);
  auto app = holoscan::make_application<OfflineCudaDinoOperatorReplayApp>();
  app->set_overrides(overrides);
  app->enable_metadata(true);

  const auto config_path = resolve_config_path(argv[0], overrides.config_path);
  if (!std::filesystem::exists(config_path)) {
    HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist", static_cast<std::string>(config_path));
    return -1;
  }

  app->config(config_path);
  app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>("greedy-scheduler"));
  app->run();
  return 0;
}
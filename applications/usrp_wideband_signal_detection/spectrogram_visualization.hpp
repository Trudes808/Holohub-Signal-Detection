#pragma once

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace holoscan::ops {

class SpectrogramToHolovizOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrogramToHolovizOp)

  SpectrogramToHolovizOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<int> output_height_;
  Parameter<int> output_width_;
  Parameter<int> channel_filter_;
  Parameter<std::string> tensor_name_;
};

class OfflinePgmReplayOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OfflinePgmReplayOp)

  OfflinePgmReplayOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<std::string> directory_;
  Parameter<double> frame_rate_;
  Parameter<bool> repeat_;
  Parameter<int> channel_filter_;
  Parameter<std::string> tensor_name_;

  std::vector<std::filesystem::path> frames_;
  size_t next_frame_index_ = 0;
  std::chrono::steady_clock::time_point next_deadline_{};
};

class RenderBufferScreenshotOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RenderBufferScreenshotOp)

  RenderBufferScreenshotOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<std::string> output_path_;
  bool saved_ = false;
};

struct OfflinePgmFrame {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

std::vector<std::filesystem::path> list_offline_pgm_frames(const std::filesystem::path& directory,
                                                           int channel_filter);

bool load_offline_pgm_frame(const std::filesystem::path& path, OfflinePgmFrame& frame);

std::vector<uint8_t> colorize_grayscale_spectrogram(const std::vector<uint8_t>& grayscale);

std::vector<HolovizOp::InputSpec> make_spectrogram_input_specs(const std::string& tensor_name);

}  // namespace holoscan::ops
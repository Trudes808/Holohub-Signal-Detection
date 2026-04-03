#pragma once

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace holoscan::ops {

struct ChannelVisualizationState;

class SpectrogramToHolovizOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrogramToHolovizOp)

  SpectrogramToHolovizOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<int> num_channels_;
  Parameter<int> history_frames_;
  Parameter<int> output_height_;
  Parameter<int> output_width_;
  Parameter<int> channel_filter_;
  Parameter<std::string> tensor_name_;
  Parameter<float> blue_limit_;
  Parameter<float> red_limit_;
  Parameter<float> overlay_alpha_;
  Parameter<double> center_frequency_hz_;
  Parameter<int> fft_size_;
  Parameter<int> dino_chunk_rows_;
  Parameter<int> dino_chunk_cols_;
  std::vector<ChannelVisualizationState> channel_states_;
};

class OfflinePgmReplayOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OfflinePgmReplayOp)

  OfflinePgmReplayOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<int> num_channels_;
  Parameter<int> history_frames_;
  Parameter<std::string> directory_;
  Parameter<std::string> mask_directory_;
  Parameter<double> frame_rate_;
  Parameter<bool> repeat_;
  Parameter<int> channel_filter_;
  Parameter<std::string> tensor_name_;
  Parameter<float> blue_limit_;
  Parameter<float> red_limit_;
  Parameter<float> overlay_alpha_;
  Parameter<double> center_frequency_hz_;
  Parameter<int> fft_size_;
  Parameter<int> dino_chunk_rows_;
  Parameter<int> dino_chunk_cols_;

  std::vector<std::filesystem::path> frames_;
  size_t next_frame_index_ = 0;
  std::chrono::steady_clock::time_point next_deadline_{};
  std::vector<ChannelVisualizationState> channel_states_;
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

struct VisualizationFrameInfo {
  int channel = -1;
  int64_t frame_number = -1;
  double center_frequency_hz = 0.0;
  double span_hz = 0.0;
  double resolution_hz = 0.0;
  int fft_size = 0;
  int dino_chunk_rows = 0;
  int dino_chunk_cols = 0;
  bool overlay_available = false;
  std::string title = "USRP WIDEBAND";
  std::string subtitle = "SPECTROGRAM";
};

struct ChannelVisualizationState {
  bool active = false;
  int history_width = 0;
  int latest_frame_height = 0;
  std::vector<uint8_t> history_grayscale;
  std::vector<float> current_psd_trace;
  std::vector<float> max_hold_trace;
  std::vector<float> density_trace;
  size_t density_frames_seen = 0;
  OfflinePgmFrame latest_mask;
  bool overlay_available = false;
  VisualizationFrameInfo info;
};

std::vector<std::filesystem::path> list_offline_pgm_frames(const std::filesystem::path& directory,
                                                           int channel_filter);

bool load_offline_pgm_frame(const std::filesystem::path& path, OfflinePgmFrame& frame);

bool parse_recorded_pgm_name(const std::string& filename,
                             const std::string& prefix,
                             int& channel,
                             uint64_t& frame_number,
                             int& rows,
                             int& cols);

std::filesystem::path find_matching_recorded_pgm(const std::filesystem::path& directory,
                                                 const std::string& prefix,
                                                 int channel,
                                                 uint64_t frame_number);

std::vector<uint8_t> colorize_grayscale_spectrogram(const std::vector<uint8_t>& grayscale,
                                                    float blue_limit,
                                                    float red_limit);

std::vector<float> compute_psd_trace(const std::vector<uint8_t>& grayscale,
                                     int width,
                                     int height);

std::vector<float> compute_density_trace(const OfflinePgmFrame* mask_frame);

std::vector<float> compute_density_trace_from_grayscale(const std::vector<uint8_t>& grayscale,
                                                        int width,
                                                        int height,
                                                        float threshold);

void update_max_hold_trace(const std::vector<float>& current_trace,
                           std::vector<float>& max_hold_trace);

void update_density_history(const std::vector<float>& current_density,
                            std::vector<float>& density_history,
                            size_t& density_frames_seen);

void append_spectrogram_history(ChannelVisualizationState& state,
                                const std::vector<uint8_t>& grayscale,
                                int width,
                                int height,
                                int history_frames);

std::vector<uint8_t> compose_visualization_rgb(const std::vector<ChannelVisualizationState>& channels,
                                               float blue_limit,
                                               float red_limit,
                                               float overlay_alpha,
                                               int& output_width,
                                               int& output_height);

std::vector<HolovizOp::InputSpec> make_spectrogram_input_specs(const std::string& tensor_name);

}  // namespace holoscan::ops
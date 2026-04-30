#pragma once

#include <cuda/std/complex>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <matx.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace holoscan::ops {

using SpectrogramComplex = cuda::std::complex<float>;
using SpectrogramTensor = matx::tensor_t<SpectrogramComplex, 2>;

struct VisualSpectrogramMessage {
  std::vector<uint8_t> pixels;
  int width = 0;
  int height = 0;
  int channel = 0;
  uint64_t frame_number = 0;
  uint64_t fft_emit_ts_ns = 0;
  uint64_t preview_enter_ts_ns = 0;
  uint64_t preview_emit_ts_ns = 0;
  double center_frequency_hz = 0.0;
  double sample_rate_hz = 0.0;
  double span_hz = 0.0;
  double resolution_hz = 0.0;
};

struct DetectorMaskMessage;

class SpectrogramPreviewOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrogramPreviewOp)

  using in_t = std::tuple<SpectrogramTensor, cudaStream_t>;
  using out_t = VisualSpectrogramMessage;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  struct PreviewTimingStats {
    uint64_t frames_seen = 0;
    double fft_to_preview_total_ms = 0.0;
    double fft_to_preview_max_ms = 0.0;
    double preview_compute_total_ms = 0.0;
    double preview_compute_max_ms = 0.0;
    double fft_to_preview_emit_total_ms = 0.0;
    double fft_to_preview_emit_max_ms = 0.0;
  };

  Parameter<int> channel_index_;
  Parameter<int> emit_every_n_;
  Parameter<int> output_width_;
  Parameter<int> output_height_;
  Parameter<float> db_floor_;
  Parameter<float> db_ceil_;
  Parameter<bool> timing_summary_enable_;
  Parameter<int> timing_summary_every_n_;
  uint64_t frames_seen_ = 0;
  cudaStream_t reduce_stream_ = nullptr;
  uint8_t* device_output_ = nullptr;
  void* pinned_output_ = nullptr;
  size_t buffer_bytes_ = 0;
  std::vector<PreviewTimingStats> timing_stats_;

  void ensure_preview_capacity(size_t required_bytes);
};

class MaskPreviewOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MaskPreviewOp)

  using in_t = DetectorMaskMessage;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  Parameter<int> emit_every_n_;
  Parameter<int> output_width_;
  Parameter<int> output_height_;
  uint64_t frames_seen_ = 0;
  cudaStream_t reduce_stream_ = nullptr;
  uint8_t* device_output_ = nullptr;
  void* pinned_output_ = nullptr;
  size_t buffer_bytes_ = 0;

  void ensure_preview_capacity(size_t required_bytes);
};

class SpectrogramPreviewStoreOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrogramPreviewStoreOp)

  SpectrogramPreviewStoreOp() = default;

  using in_t = VisualSpectrogramMessage;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;
};

class MaskPreviewStoreOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MaskPreviewStoreOp)

  MaskPreviewStoreOp() = default;

  using in_t = DetectorMaskMessage;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;
};

struct ChannelVisualizationState;

class SpectrogramToHolovizOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrogramToHolovizOp)

  SpectrogramToHolovizOp() = default;

  // void setup(OperatorSpec& spec) override;
  // void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
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
  Parameter<bool> overlay_enable_;
  Parameter<std::string> detector_label_;
  Parameter<double> center_frequency_hz_;
  Parameter<double> span_hz_;
  Parameter<int> fft_size_;
  Parameter<int> dino_chunk_rows_;
  Parameter<int> dino_chunk_cols_;
  Parameter<int> display_time_rows_;
  Parameter<int> display_freq_bins_;
  Parameter<int> history_memory_budget_mb_;
  Parameter<int> rows_per_frame_;
  Parameter<int> mask_frame_offset_;
  Parameter<int> render_every_n_frames_;
  Parameter<bool> timing_summary_enable_;
  Parameter<int> timing_summary_every_n_;
  Parameter<std::shared_ptr<BooleanCondition>> shutdown_scheduling_term_;
  Parameter<float> db_floor_;
  Parameter<float> db_ceil_;
  Parameter<int> row_average_n_;

  struct VisualChannelResources {
    cudaStream_t stream = nullptr;
    uint8_t* device_grayscale_buffer = nullptr;
    void* pinned_grayscale_buffer = nullptr;
    uint8_t* device_mask_buffer = nullptr;
    void* pinned_mask_buffer = nullptr;
    size_t grayscale_buffer_bytes = 0;
  };

  struct PendingChannelFrame {
    bool pending = false;
    bool compose_requested = false;
    int channel = -1;
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int display_time_rows = 0;
    uint64_t frame_number = 0;
    double center_frequency_hz = 0.0;
    double span_hz = 0.0;
    double resolution_hz = 0.0;
  };

  struct VisualTimingStats {
    uint64_t frames_seen = 0;
    uint64_t frames_processed = 0;
    uint64_t frames_rendered = 0;
    uint64_t dropped_vis_stream_busy = 0;
    uint64_t dropped_render_queue_busy = 0;
    uint64_t render_skipped_by_cadence = 0;
    uint64_t masks_received = 0;
    uint64_t masks_backfilled = 0;
    uint64_t masks_deferred = 0;
    uint64_t masks_pending_peak = 0;
    double sync_total_ms = 0.0;
    double sync_max_ms = 0.0;
    double reduce_total_ms = 0.0;
    double reduce_max_ms = 0.0;
    double history_total_ms = 0.0;
    double history_max_ms = 0.0;
    double compose_total_ms = 0.0;
    double compose_max_ms = 0.0;
    double render_total_ms = 0.0;
    double render_max_ms = 0.0;
    double fft_to_visualizer_total_ms = 0.0;
    double fft_to_visualizer_max_ms = 0.0;
    double preview_to_visualizer_total_ms = 0.0;
    double preview_to_visualizer_max_ms = 0.0;
  };

  struct PerChannelVisualTimingStats {
    uint64_t frames_processed = 0;
    uint64_t last_frame_number = 0;
    double fft_to_visualizer_total_ms = 0.0;
    double fft_to_visualizer_max_ms = 0.0;
    double preview_to_visualizer_total_ms = 0.0;
    double preview_to_visualizer_max_ms = 0.0;
  };

  std::vector<ChannelVisualizationState> channel_states_;
  std::mutex channel_states_mutex_;
  std::vector<VisualChannelResources> channel_resources_;
  std::vector<PendingChannelFrame> pending_channel_frames_;
  uint64_t dropped_frames_ = 0;
  uint64_t total_frames_ = 0;
  std::mutex timing_mutex_;
  VisualTimingStats timing_stats_;
  std::vector<PerChannelVisualTimingStats> per_channel_timing_stats_;

  // Background render thread — keeps operator thread free
  std::thread render_thread_;
  std::mutex render_mutex_;
  std::condition_variable render_cv_;
  bool render_work_pending_ = false;
  bool render_stop_ = false;

  // Pending composed frame — background thread fills this,
  // operator thread emits it on the next tick
  std::mutex composed_mutex_;
  std::vector<uint8_t> pending_composed_;
  int pending_composed_width_ = 0;
  int pending_composed_height_ = 0;
  bool pending_composed_ready_ = false;
  std::atomic<bool> history_budget_warning_emitted_{false};
  std::atomic<bool> channel_filter_override_warning_emitted_{false};
  std::vector<uint64_t> latest_rendered_frame_numbers_;

  void ensure_channel_resource_capacity(size_t channel_index, size_t required_bytes);

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
  Parameter<bool> overlay_enable_;
  Parameter<std::string> detector_label_;
  Parameter<double> center_frequency_hz_;
  Parameter<double> span_hz_;
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

struct DetectorMaskMessage {
  std::vector<uint8_t> pixels;
  std::shared_ptr<uint8_t> device_pixels;
  int width = 0;
  int height = 0;
  int channel = 0;
  uint64_t frame_number = 0;
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
  std::string detector_label = "Dinov3";
};

struct VisualizationRect {
  float x = 0.0f;
  float y = 0.0f;
  float width = 0.0f;
  float height = 0.0f;
};

struct VisualizationChannelUiState {
  bool active = false;
  int channel = -1;
  VisualizationRect header_rect;
  VisualizationRect psd_rect;
  VisualizationRect waterfall_rect;
  VisualizationRect mask_rect;
  VisualizationRect confidence_rect;
  double center_frequency_hz = 0.0;
  double span_hz = 0.0;
  double resolution_hz = 0.0;
  double seconds_per_time_bin = 0.0;
  int fft_size = 0;
  int history_rows = 0;
  int dino_chunk_rows = 0;
  int dino_chunk_cols = 0;
  float confidence = 0.0f;
  bool overlay_available = false;
  std::string detector_label = "Dinov3";
};

struct VisualizationUiState {
  int canvas_width = 0;
  int canvas_height = 0;
  bool overlay_enabled = true;
  float blue_limit = 0.0f;
  float red_limit = 0.0f;
  uint64_t dropped_frames = 0;
  uint64_t total_frames = 0;
  std::string title = "USRP WIDEBAND";
  std::string subtitle = "REAL TIME SIGNAL DETECTION";
  std::string detector_label = "Dinov3";
  VisualizationRect header_rect;
  VisualizationRect content_rect;
  VisualizationRect sidebar_rect;
  std::vector<VisualizationChannelUiState> channels;
};

struct ChannelVisualizationState {
  bool active = false;
  int history_width = 0;
  int latest_frame_height = 0;
  int history_capacity_rows = 0;
  int history_valid_rows = 0;
  int history_write_row = 0;
  std::vector<uint8_t> history_grayscale;
  std::vector<uint8_t> history_mask;
  std::vector<int64_t> history_row_frame_numbers;
  std::vector<int> history_row_indices_within_frame;
  std::vector<float> current_psd_trace;
  std::vector<float> max_hold_trace;
  std::vector<float> density_trace;
  size_t density_frames_seen = 0;
  OfflinePgmFrame latest_mask;
  int64_t latest_mask_frame_number = -1;
  bool overlay_available = false;
  VisualizationFrameInfo info;
  std::unordered_map<int64_t, DetectorMaskMessage> pending_masks;

  // FFT row accumulator for averaging instead of dropping
  std::vector<float> row_accumulator;
  int row_accumulator_count = 0;
  static constexpr int kRowAverageN = 4;  // average 4 frames per display row
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

std::vector<uint8_t> reduce_mask_to_history_rows(const OfflinePgmFrame& mask_frame,
                                                 int dst_width,
                                                 int dst_rows);

bool patch_history_mask_for_frame(ChannelVisualizationState& state,
                                  int64_t frame_number,
                                  const std::vector<uint8_t>& mask_rows,
                                  int width);

void append_spectrogram_history(ChannelVisualizationState& state,
                                const std::vector<uint8_t>& grayscale,
                                int width,
                                int height,
                                int max_rows);

// std::vector<uint8_t> compose_visualization_rgb(const std::vector<ChannelVisualizationState>& channels,
//                                                float blue_limit,
//                                                float red_limit,
//                                                float overlay_alpha,
//                                                int& output_width,
//                                                int& output_height);

std::vector<uint8_t> compose_visualization_rgb(const std::vector<ChannelVisualizationState>& channels,
                                               float blue_limit,
                                               float red_limit,
                                               float overlay_alpha,
                                               bool overlay_enabled,
                                               int panel_width,
                                               int panel_height,
                                               int& output_width,
                                               int& output_height,
                                               uint64_t dropped_frames = 0,
                                               uint64_t total_frames = 0);

std::vector<HolovizOp::InputSpec> make_spectrogram_input_specs(const std::string& tensor_name);

void initialize_visualization_overlay_state(bool enabled);

void set_visualization_overlay_enabled(bool enabled);

bool visualization_overlay_enabled();

void set_visualization_full_ui_enabled(bool enabled);

bool visualization_full_ui_enabled();

void update_visualization_ui_state(const VisualizationUiState& state);

VisualizationUiState visualization_ui_state_snapshot();

void render_visualization_ui_overlay();

}  // namespace holoscan::ops
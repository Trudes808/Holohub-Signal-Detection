// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "mask_replay_detector.hpp"

#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <matx.h>

// DetectorMaskMessage lives in the app header (same include cuda_dino_detector uses).
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

namespace holoscan::ops {

// Input wire type = the spectrogram tuple (identical to CudaDinoDetector's input).
// We receive it only to advance the port + pick up message metadata; the tensor
// payload is unused.  TODO(lab-admin): this mirrors cuda_dino_types.hpp's alias
// (cuda_dino receives the same wire message), but confirm the complex element type
// matches SpectrogramComplex if the type check fails to compile.
using mask_replay_complex = cuda::std::complex<float>;
using mask_replay_in_t = std::tuple<matx::tensor_t<mask_replay_complex, 2>, cudaStream_t>;

namespace {

// Minimal reader for a C-order uint8 2-D .npy (mask_ch{c}_f{N}_{H}x{W}.npy).
// Fills rows(=H, time), cols(=W, freq), data (row-major H*W). Returns false on any mismatch.
bool load_npy_u8(const std::filesystem::path& path, int& rows, int& cols, std::vector<uint8_t>& data) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[6] = {0};
  f.read(magic, 6);
  if (f.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
  uint8_t ver_major = 0, ver_minor = 0;
  f.read(reinterpret_cast<char*>(&ver_major), 1);
  f.read(reinterpret_cast<char*>(&ver_minor), 1);
  uint32_t header_len = 0;
  if (ver_major == 1) {
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    header_len = hl;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  if (static_cast<uint32_t>(f.gcount()) != header_len) return false;
  if (header.find("'fortran_order': True") != std::string::npos) {
    std::fprintf(stderr, "[mask_replay_detector] ERROR: %s is fortran_order (unsupported)\n",
                 path.c_str());
    return false;
  }
  if (header.find("u1") == std::string::npos && header.find("b1") == std::string::npos) {
    std::fprintf(stderr, "[mask_replay_detector] WARN: %s descr is not uint8; reading raw bytes anyway\n",
                 path.c_str());
  }
  std::smatch m;
  const std::regex shape_re("'shape':\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)");
  if (!std::regex_search(header, m, shape_re)) return false;
  rows = std::stoi(m[1].str());
  cols = std::stoi(m[2].str());
  const size_t n = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  data.resize(n);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n));
  return static_cast<size_t>(f.gcount()) == n;
}

// Find mask_ch{channel}_f{frame}_*.npy (frame delimited by the trailing '_', so f10 != f100).
std::filesystem::path find_mask_file(const std::string& dir, int channel, uint64_t frame) {
  namespace fs = std::filesystem;
  const std::string prefix =
      "mask_ch" + std::to_string(channel) + "_f" + std::to_string(frame) + "_";
  std::error_code ec;
  if (dir.empty() || !fs::is_directory(dir, ec)) return {};
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    if (ec) break;
    const std::string name = entry.path().filename().string();
    if (name.rfind(prefix, 0) == 0 && entry.path().extension() == ".npy") return entry.path();
  }
  return {};
}

}  // namespace

void MaskReplayDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<mask_replay_in_t>("in");
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);
  spec.param(mask_dir_, "mask_dir", "Mask directory",
             "Directory of precomputed masks (mask_ch{c}_f{N}_{H}x{W}.npy) to replay.", std::string{});
  spec.param(channel_, "channel", "Channel", "Channel index used in the mask filename.", 0);
  spec.param(emit_stride_, "emit_stride", "Emit stride",
             "Emit every Nth frame (match the run that produced the masks).", 1);
  spec.param(num_channels_, "num_channels", "Num channels", "Number of detector channels.", 1);
}

void MaskReplayDetector::compute(holoscan::InputContext& op_input,
                                 holoscan::OutputContext& op_output,
                                 holoscan::ExecutionContext& /*context*/) {
  // Consume the input to advance the port + pick up metadata; tensor payload unused.
  auto maybe_input = op_input.receive<mask_replay_in_t>("in");
  if (!maybe_input) return;

  auto meta = metadata();
  const int channel = channel_.get();

  // frame_number: prefer the FFT's per-input counter (== the N in the mask filename).
  uint64_t frame_number = 0;
  if (meta && meta->has_key("fft_emitted_frame_number")) {
    frame_number = meta->get<uint64_t>("fft_emitted_frame_number", frame_counter_ + 1);
    if (frame_number == 0) frame_number = frame_counter_ + 1;
    frame_counter_ = std::max(frame_counter_, frame_number);
  } else {
    frame_number = ++frame_counter_;
  }

  if (!startup_log_emitted_) {
    startup_log_emitted_ = true;
    std::fprintf(stderr,
                 "[mask_replay_detector] INFO: replaying masks from '%s' channel=%d emit_stride=%d\n",
                 mask_dir_.get().c_str(), channel, emit_stride_.get());
  }

  // Drain / partial frames carry no real mask -> do not emit (mirrors cuda_dino).
  if (meta && (meta->get<bool>("offline_source_drain_frame", false) ||
               meta->get<bool>("chdr_partial_batch", false))) {
    if (meta) meta->set("mask_replay_mask_emitted", false);
    return;
  }
  const int stride = std::max(1, emit_stride_.get());
  if ((frame_number % static_cast<uint64_t>(stride)) != 0) {
    if (meta) meta->set("mask_replay_mask_emitted", false);
    return;
  }

  int rows = 0, cols = 0;
  std::vector<uint8_t> pixels;
  const std::filesystem::path mask_path = find_mask_file(mask_dir_.get(), channel, frame_number);
  bool loaded = !mask_path.empty() && load_npy_u8(mask_path, rows, cols, pixels);

  if (!loaded) {
    // Missing/unreadable: emit an all-zero mask of the last-known geometry so the snipper still
    // sees the frame as "no detection". Skip if we have no geometry yet.
    ++missing_mask_count_;
    if (missing_mask_count_ <= 20) {
      std::fprintf(stderr, "[mask_replay_detector] WARN: no mask for frame %llu in '%s' (%s)\n",
                   static_cast<unsigned long long>(frame_number), mask_dir_.get().c_str(),
                   (last_rows_ > 0 ? "emitting zero mask" : "skipping - no geometry yet"));
    }
    if (last_rows_ <= 0 || last_cols_ <= 0) {
      if (meta) meta->set("mask_replay_mask_emitted", false);
      return;
    }
    rows = last_rows_;
    cols = last_cols_;
    pixels.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  } else {
    last_rows_ = rows;
    last_cols_ = cols;
  }

  const uint16_t channel_number =
      meta ? meta->get<uint16_t>("channel_number", static_cast<uint16_t>(channel))
           : static_cast<uint16_t>(channel);

  holoscan::ops::DetectorMaskMessage mask_msg;
  mask_msg.pixels = std::move(pixels);   // host mask; device_pixels null -> snipper reads pixels
  mask_msg.width = cols;                 // freq bins
  mask_msg.height = rows;                // time rows
  mask_msg.channel = static_cast<int>(channel_number);
  mask_msg.frame_number = frame_number;
  if (meta) {
    mask_msg.file_offset_complex = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
    mask_msg.data_end_complex = meta->get<uint64_t>("offline_source_data_end_complex", 0);
    mask_msg.frame_end_complex = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
    mask_msg.complex_samples_read = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
    mask_msg.complex_samples_padded = meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
  }
  op_output.emit(mask_msg, "mask_out");
  if (meta) meta->set("mask_replay_mask_emitted", true);
}

}  // namespace holoscan::ops

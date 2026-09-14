// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "finetuned_dino_mask_dump.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

namespace holoscan::ops {

namespace {

// Bounded queue depth. The writer streams ~a few MB per mask; a small buffer absorbs jitter, and
// if the writer ever falls behind we drop (and count) rather than stall the detector's data path.
constexpr size_t kMaxQueued = 48;

// Write a 2-D uint8 array as a NumPy v1.0 .npy (dtype '|u1', C-order). Mirrors the offline eval's
// writer so mask_eval_metrics.load_mask_any reads dumped and offline masks identically.
void write_npy_u8(const std::string& path, const uint8_t* data, size_t rows, size_t cols) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "[mask_dump] failed to open %s for write\n", path.c_str());
    return;
  }
  std::string hdr = "{'descr': '|u1', 'fortran_order': False, 'shape': (" +
                    std::to_string(rows) + ", " + std::to_string(cols) + "), }";
  // preamble = 6 (magic) + 2 (version) + 2 (header-len field) = 10 bytes.
  const size_t total = 10 + hdr.size() + 1 /*newline*/;
  const size_t pad = (64 - (total % 64)) % 64;
  hdr.append(pad, ' ');
  hdr.push_back('\n');
  const uint16_t hlen = static_cast<uint16_t>(hdr.size());  // host is little-endian (ARM64)

  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  const char ver[2] = {1, 0};
  f.write(magic, 6);
  f.write(ver, 2);
  f.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
  f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(rows * cols));
}

// Same, for a 2-D float32 array (dtype '<f4') -- the RT input spectrogram, for mask-on-spectrogram overlays.
void write_npy_f32(const std::string& path, const float* data, size_t rows, size_t cols) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "[mask_dump] failed to open %s for write\n", path.c_str());
    return;
  }
  std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                    std::to_string(rows) + ", " + std::to_string(cols) + "), }";
  const size_t total = 10 + hdr.size() + 1;
  const size_t pad = (64 - (total % 64)) % 64;
  hdr.append(pad, ' ');
  hdr.push_back('\n');
  const uint16_t hlen = static_cast<uint16_t>(hdr.size());
  const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  const char ver[2] = {1, 0};
  f.write(magic, 6);
  f.write(ver, 2);
  f.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
  f.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(rows * cols * sizeof(float)));
}

}  // namespace

struct Job {
  int channel = 0;
  uint64_t frame = 0;
  int rows = 0;
  int cols = 0;
  std::vector<uint8_t> data;
  std::vector<float> spec;   // optional rows*cols float32 input spectrogram (empty = not dumped)
};

struct MaskDumpWriter::Impl {
  std::string dir;
  int max_frames = 0;

  std::atomic<int> submitted{0};
  std::atomic<int> written{0};
  std::atomic<int> dropped{0};

  std::deque<Job> queue;
  std::mutex mtx;
  std::condition_variable cv;
  bool stopping = false;
  std::thread worker;

  std::ofstream manifest;
  bool manifest_header_written = false;

  void run() {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [&] { return stopping || !queue.empty(); });
        if (queue.empty()) {
          if (stopping) return;
          continue;
        }
        job = std::move(queue.front());
        queue.pop_front();
      }
      const int seq = written.load();
      char fname[128];
      std::snprintf(fname, sizeof(fname), "mask_ch%d_%06d.npy", job.channel, seq);
      write_npy_u8(dir + "/" + fname, job.data.data(),
                   static_cast<size_t>(job.rows), static_cast<size_t>(job.cols));
      char sname[128] = "";
      if (!job.spec.empty()) {
        std::snprintf(sname, sizeof(sname), "spec_ch%d_%06d.npy", job.channel, seq);
        write_npy_f32(dir + "/" + sname, job.spec.data(),
                      static_cast<size_t>(job.rows), static_cast<size_t>(job.cols));
      }
      if (manifest.is_open()) {
        manifest << seq << ',' << job.channel << ',' << job.frame << ',' << job.rows << ','
                 << job.cols << ',' << fname << ',' << sname << '\n';
        manifest.flush();
      }
      written.fetch_add(1);
    }
  }
};

MaskDumpWriter::MaskDumpWriter() = default;

MaskDumpWriter::~MaskDumpWriter() { flush_and_stop(); }

void MaskDumpWriter::configure(const std::string& dir, int max_frames) {
  if (dir.empty()) {
    enabled_ = false;
    return;
  }
  impl_ = std::make_unique<Impl>();
  impl_->dir = dir;
  impl_->max_frames = max_frames;

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    std::fprintf(stderr, "[mask_dump] could not create %s: %s -- dump disabled\n", dir.c_str(),
                 ec.message().c_str());
    impl_.reset();
    enabled_ = false;
    return;
  }

  impl_->manifest.open(dir + "/mask_dump_manifest.csv", std::ios::out | std::ios::trunc);
  if (impl_->manifest.is_open()) {
    impl_->manifest << "seq,channel,frame_number,rows,cols,mask_npy,spec_npy\n";
    impl_->manifest.flush();
    impl_->manifest_header_written = true;
  }

  impl_->worker = std::thread([this] { impl_->run(); });
  enabled_ = true;
  std::fprintf(stderr, "[mask_dump] ENABLED -> %s (max_frames=%d)\n", dir.c_str(), max_frames);
}

bool MaskDumpWriter::wants_more() const {
  if (!enabled_ || !impl_) return false;
  if (impl_->max_frames <= 0) return true;
  return impl_->submitted.load() < impl_->max_frames;
}

void MaskDumpWriter::submit(int channel, uint64_t frame_number, int rows, int cols,
                            const uint8_t* host_mask, const float* host_spec) {
  if (!enabled_ || !impl_ || host_mask == nullptr || rows <= 0 || cols <= 0) return;
  if (impl_->max_frames > 0 && impl_->submitted.load() >= impl_->max_frames) return;

  Job job;
  job.channel = channel;
  job.frame = frame_number;
  job.rows = rows;
  job.cols = cols;
  job.data.assign(host_mask, host_mask + static_cast<size_t>(rows) * cols);
  if (host_spec != nullptr) {
    job.spec.assign(host_spec, host_spec + static_cast<size_t>(rows) * cols);
  }

  {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    if (impl_->queue.size() >= kMaxQueued) {
      impl_->dropped.fetch_add(1);
      return;  // never block the data path
    }
    impl_->queue.push_back(std::move(job));
    impl_->submitted.fetch_add(1);
  }
  impl_->cv.notify_one();
}

void MaskDumpWriter::flush_and_stop() {
  if (!impl_) return;
  {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    impl_->stopping = true;
  }
  impl_->cv.notify_all();
  if (impl_->worker.joinable()) impl_->worker.join();
  if (impl_->manifest.is_open()) impl_->manifest.close();
  std::fprintf(stderr, "[mask_dump] stopped: %d written, %d submitted, %d dropped -> %s\n",
               impl_->written.load(), impl_->submitted.load(), impl_->dropped.load(),
               impl_->dir.c_str());
  impl_.reset();
  enabled_ = false;
}

}  // namespace holoscan::ops
